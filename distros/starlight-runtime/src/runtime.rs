// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 SeekStarAI and StarLight Contributors

//! CLI and runtime assembly.
//!
//! The runtime is the composition root from ARCH-0005 §5.5 and ARCH-0007 §3.4:
//! it parses configuration, compiles the fixture's prefill/decode programs with
//! `sl-graph`, uploads weights through a `CpuDevice` and drives
//! `sl_core::Engine`. It contains no kernels, no tokenizer and no HTTP.
//!
//! # Exit codes
//!
//! | Code | Meaning |
//! | --- | --- |
//! | 0 | success |
//! | 1 | unexpected output I/O failure |
//! | 2 | configuration or command-line error |
//! | 3 | graph compilation or model/engine build error |
//! | 4 | device error |
//! | 5 | not implemented (`serve`) |

use std::fmt;
use std::io::{self, Write};
use std::path::PathBuf;

use serde::Serialize;
use sl_core::{
    BuildError, CompiledProgram, Engine, EngineConfig as CoreEngineConfig, StepError, StepInput,
    StepMode,
};
use sl_device::{Device, DeviceError};
use sl_ir::{
    BackendTag, BufferId, CommandBuffer, DTypeSet, DeviceCapabilities, OpCode, OpSet,
    ShapeConstraints, ValueId, IR_VERSION,
};
use starlight_backend_cpu::CpuDevice;

use crate::config::{Config, ConfigError};
use crate::fixture::{Fixture, MAX_SEQ, VOCAB};

/// Exit code for success.
pub const EXIT_OK: i32 = 0;
/// Exit code for an unexpected output failure.
pub const EXIT_OUTPUT: i32 = 1;
/// Exit code for configuration and command-line errors.
pub const EXIT_USAGE: i32 = 2;
/// Exit code for graph compilation and model/engine build errors.
pub const EXIT_COMPILE: i32 = 3;
/// Exit code for device errors.
pub const EXIT_DEVICE: i32 = 4;
/// Exit code for commands that are not implemented yet.
pub const EXIT_UNIMPLEMENTED: i32 = 5;

/// Buffer alignment requested from the device; matches `CpuDevice`'s contract.
pub const BUFFER_ALIGNMENT: usize = 64;

/// Capabilities reported by `describe`.
///
/// The runtime only accepts token IDs until `sl-tokenizer` and `sl-daemon`
/// exist, so it does not claim text or streaming support.
pub const CAPABILITIES: &[&str] = &["token-ids"];

/// One parsed CLI invocation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Command {
    /// Print the ARCH-0005 §5.2 describe document.
    Describe {
        /// Path to the JSON configuration.
        config: PathBuf,
    },
    /// Compile the fixture and build an engine without stepping it.
    Validate {
        /// Path to the JSON configuration.
        config: PathBuf,
    },
    /// Run one prefill step and print logits.
    Run {
        /// Path to the JSON configuration.
        config: PathBuf,
        /// Token ids for the prefill step.
        input_ids: Vec<u32>,
    },
    /// Return the explicit "not implemented" error.
    Serve {
        /// Path to the JSON configuration.
        config: PathBuf,
    },
    /// Print the crate version and the IR version.
    Version,
    /// Print usage.
    Help,
}

/// Failure while parsing command-line arguments.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CliError(String);

impl CliError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for CliError {}

/// Failure while executing a [`Command`].
#[derive(Debug)]
pub enum RuntimeError {
    /// Configuration could not be loaded or is invalid.
    Config(ConfigError),
    /// The command's arguments are invalid for the loaded configuration.
    Usage(String),
    /// `sl-graph` rejected the fixture or failed to compile it.
    Compile(String),
    /// `sl-core` rejected the engine configuration.
    Model(String),
    /// The device rejected an allocation, upload or step.
    Device(String),
    /// The command is recognized but not implemented.
    Unsupported(String),
    /// Writing the command output failed.
    Output(String),
}

impl RuntimeError {
    /// Returns the process exit code for this error.
    pub const fn exit_code(&self) -> i32 {
        match self {
            Self::Config(_) | Self::Usage(_) => EXIT_USAGE,
            Self::Compile(_) | Self::Model(_) => EXIT_COMPILE,
            Self::Device(_) => EXIT_DEVICE,
            Self::Unsupported(_) => EXIT_UNIMPLEMENTED,
            Self::Output(_) => EXIT_OUTPUT,
        }
    }
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(error) => error.fmt(formatter),
            Self::Usage(message) => formatter.write_str(message),
            Self::Compile(message) => write!(formatter, "graph compilation failed: {message}"),
            Self::Model(message) => write!(formatter, "engine build failed: {message}"),
            Self::Device(message) => write!(formatter, "device error: {message}"),
            Self::Unsupported(message) => write!(formatter, "unsupported: {message}"),
            Self::Output(message) => write!(formatter, "failed to write output: {message}"),
        }
    }
}

impl std::error::Error for RuntimeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Config(error) => Some(error),
            _ => None,
        }
    }
}

/// The ARCH-0005 §5.2 describe document.
///
/// Field names and meanings are part of the runtime protocol. New fields may be
/// added; existing ones never change meaning.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Describe {
    /// Engine implementation, always `starlight`.
    pub engine: &'static str,
    /// Version of the engine implementation.
    pub engine_version: &'static str,
    /// IR version the runtime was built against.
    pub ir_version: u32,
    /// Version of this runtime binary.
    pub runtime_version: &'static str,
    /// Selected backend.
    pub backend: String,
    /// Architectures the configured model declares.
    pub architectures: Vec<String>,
    /// Maximum context length from the configuration.
    pub max_context: u32,
    /// Maximum batch size from the configuration.
    pub max_batch: u32,
    /// Quantization of the loaded model; the fixture is uncompressed.
    pub quantization: &'static str,
    /// Features this runtime implements.
    pub capabilities: Vec<String>,
}

impl Describe {
    /// Builds the describe document from configuration, without loading
    /// weights.
    pub fn from_config(config: &Config) -> Self {
        Self {
            engine: "starlight",
            engine_version: crate::VERSION,
            ir_version: IR_VERSION,
            runtime_version: crate::VERSION,
            backend: config.engine.backend.clone(),
            architectures: vec![config.model.architecture.clone()],
            max_context: config.engine.max_context,
            max_batch: config.engine.max_batch,
            quantization: "none",
            capabilities: CAPABILITIES.iter().map(|item| (*item).to_owned()).collect(),
        }
    }
}

/// Returns the usage text printed by `--help`.
pub fn usage() -> String {
    format!(
        "starlight-runtime {}\n\
         usage:\n  \
         starlight-runtime describe --config <path>\n  \
         starlight-runtime validate --config <path>\n  \
         starlight-runtime run --config <path> --input-ids <id,id,...>\n  \
         starlight-runtime serve --config <path>\n  \
         starlight-runtime version",
        crate::VERSION
    )
}

/// Parses command-line arguments, excluding the program name.
///
/// Supports both `--flag value` and `--flag=value`.
///
/// # Errors
///
/// Returns [`CliError`] for an unknown command, a missing or duplicate flag, a
/// positional argument, or an unparsable `--input-ids` list.
pub fn parse_args<I>(args: I) -> Result<Command, CliError>
where
    I: IntoIterator<Item = String>,
{
    let args: Vec<String> = args.into_iter().collect();
    let Some(name) = args.first() else {
        return Err(CliError::new(
            "missing command; expected describe, validate, run, serve or version",
        ));
    };
    let rest = &args[1..];

    match name.as_str() {
        "describe" => {
            let flags = parse_flags(rest, &["--config"])?;
            Ok(Command::Describe {
                config: PathBuf::from(require_flag(&flags, "--config")?),
            })
        }
        "validate" => {
            let flags = parse_flags(rest, &["--config"])?;
            Ok(Command::Validate {
                config: PathBuf::from(require_flag(&flags, "--config")?),
            })
        }
        "run" => {
            let flags = parse_flags(rest, &["--config", "--input-ids"])?;
            let config = PathBuf::from(require_flag(&flags, "--config")?);
            let raw = require_flag(&flags, "--input-ids")?;
            Ok(Command::Run {
                config,
                input_ids: parse_input_ids(&raw)?,
            })
        }
        "serve" => {
            let flags = parse_flags(rest, &["--config"])?;
            Ok(Command::Serve {
                config: PathBuf::from(require_flag(&flags, "--config")?),
            })
        }
        "version" if rest.is_empty() => Ok(Command::Version),
        "version" => Err(CliError::new("version takes no arguments")),
        "help" | "--help" | "-h" if rest.is_empty() => Ok(Command::Help),
        "help" | "--help" | "-h" => Err(CliError::new("help takes no arguments")),
        other => Err(CliError::new(format!(
            "unknown command {other:?}; expected describe, validate, run, serve or version"
        ))),
    }
}

fn parse_flags(
    args: &[String],
    allowed: &[&'static str],
) -> Result<Vec<(&'static str, String)>, CliError> {
    let mut flags: Vec<(&'static str, String)> = Vec::new();
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        let Some(body) = argument.strip_prefix("--") else {
            return Err(CliError::new(format!(
                "unexpected argument {argument:?}; flags must start with --"
            )));
        };
        let (name, inline) = match body.split_once('=') {
            Some((name, value)) => (name, Some(value.to_owned())),
            None => (body, None),
        };
        let Some(&flag) = allowed
            .iter()
            .find(|flag| flag.trim_start_matches('-') == name)
        else {
            return Err(CliError::new(format!(
                "unknown flag --{name}; allowed flags: {}",
                allowed.join(", ")
            )));
        };
        if flags.iter().any(|(seen, _)| *seen == flag) {
            return Err(CliError::new(format!("flag {flag} given more than once")));
        }
        let value = match inline {
            Some(value) => value,
            None => {
                index += 1;
                args.get(index)
                    .cloned()
                    .ok_or_else(|| CliError::new(format!("flag {flag} requires a value")))?
            }
        };
        if value.is_empty() {
            return Err(CliError::new(format!("flag {flag} requires a value")));
        }
        flags.push((flag, value));
        index += 1;
    }
    Ok(flags)
}

fn require_flag(flags: &[(&'static str, String)], name: &'static str) -> Result<String, CliError> {
    flags
        .iter()
        .find(|(flag, _)| *flag == name)
        .map(|(_, value)| value.clone())
        .ok_or_else(|| CliError::new(format!("missing required flag {name}")))
}

/// Parses a comma-separated list of `u32` token ids.
///
/// # Errors
///
/// Returns [`CliError`] when the list is empty, contains an empty element or
/// contains a non-numeric or out-of-range element.
pub fn parse_input_ids(raw: &str) -> Result<Vec<u32>, CliError> {
    if raw.trim().is_empty() {
        return Err(CliError::new("--input-ids must contain at least one id"));
    }
    let mut ids = Vec::new();
    for part in raw.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            return Err(CliError::new(format!(
                "invalid --input-ids {raw:?}: empty element"
            )));
        }
        let id = trimmed.parse::<u32>().map_err(|_| {
            CliError::new(format!(
                "invalid --input-ids {raw:?}: {trimmed:?} is not a u32"
            ))
        })?;
        ids.push(id);
    }
    Ok(ids)
}

/// Process entry point used by `main`; returns the exit code.
pub fn entry<I>(args: I) -> i32
where
    I: IntoIterator<Item = String>,
{
    let command = match parse_args(args) {
        Ok(command) => command,
        Err(error) => {
            eprintln!("error: {error}");
            eprintln!("run `starlight-runtime --help` for usage");
            return EXIT_USAGE;
        }
    };

    let stdout = io::stdout();
    let mut out = stdout.lock();
    let result = dispatch(&command, &mut out);
    let flush = out.flush();
    drop(out);

    match result {
        Ok(()) => match flush {
            Ok(()) => EXIT_OK,
            Err(error) => {
                eprintln!("error: failed to write output: {error}");
                EXIT_OUTPUT
            }
        },
        Err(error) => {
            eprintln!("error: {error}");
            error.exit_code()
        }
    }
}

/// Executes one command, writing its output to `out`.
///
/// # Errors
///
/// Returns [`RuntimeError`] with a stable [`RuntimeError::exit_code`].
pub fn dispatch<W: Write>(command: &Command, out: &mut W) -> Result<(), RuntimeError> {
    match command {
        Command::Describe { config } => {
            let config = load_config(config)?;
            let describe = Describe::from_config(&config);
            let json = serde_json::to_string_pretty(&describe)
                .expect("Describe only contains serializable fields");
            write_line(out, &json)
        }
        Command::Validate { config } => {
            let config = load_config(config)?;
            validate_engine(&config)?;
            write_line(out, "ok")
        }
        Command::Run { config, input_ids } => {
            let config = load_config(config)?;
            let logits = run_prefill(&config, input_ids)?;
            for logit in logits {
                write_line(out, &format!("{logit}"))?;
            }
            Ok(())
        }
        Command::Serve { config } => {
            let config = load_config(config)?;
            serve(&config)
        }
        Command::Version => {
            write_line(out, &format!("starlight-runtime {}", crate::VERSION))?;
            write_line(out, &format!("ir_version {IR_VERSION}"))
        }
        Command::Help => write_line(out, &usage()),
    }
}

/// The `serve` command.
///
/// HTTP and the OpenAI surface live in `sl-daemon` (ARCH-0005 §4). Until that
/// crate exists, `serve` reports an explicit [`RuntimeError::Unsupported`]
/// instead of starting a private HTTP server.
///
/// # Errors
///
/// Always returns [`RuntimeError::Unsupported`].
pub fn serve(config: &Config) -> Result<(), RuntimeError> {
    Err(RuntimeError::Unsupported(format!(
        "serve is not implemented: sl-daemon is not ready (requested bind {}:{})",
        config.server.bind, config.server.port
    )))
}

/// Returns the device capabilities used to compile the fixture.
///
/// ARCH-0007 freezes `sl-graph::compile` but not capability discovery on
/// `CpuDevice`, so the composition root declares exactly what the fixture
/// needs: the four core operators it uses, `F32`/`U32` data and 64-byte
/// alignment.
pub fn cpu_capabilities() -> DeviceCapabilities {
    DeviceCapabilities {
        backend: BackendTag("cpu"),
        ir_version: IR_VERSION,
        core_ops: OpSet::new(vec![
            OpCode::Gather,
            OpCode::RmsNorm,
            OpCode::MatMulNT,
            OpCode::LastToken,
        ]),
        composite_ops: OpSet::new(Vec::new()),
        dtypes: DTypeSet::ONLY_F32.union(DTypeSet::ONLY_U32),
        min_buffer_align: BUFFER_ALIGNMENT as u32,
        max_scratchpad_bytes: 1 << 30,
        max_value_bytes: 1 << 28,
        constraints: ShapeConstraints::default(),
    }
}

fn load_config(path: &std::path::Path) -> Result<Config, RuntimeError> {
    Config::load(path).map_err(RuntimeError::Config)
}

fn write_line<W: Write>(out: &mut W, line: &str) -> Result<(), RuntimeError> {
    writeln!(out, "{line}").map_err(|error| RuntimeError::Output(error.to_string()))
}

/// Validates the fixture end to end without stepping the engine.
fn validate_engine(config: &Config) -> Result<(), RuntimeError> {
    let fixture = Fixture::build();
    let capabilities = cpu_capabilities();
    let seq = config.engine.max_context.clamp(1, 2);
    let prefill = compile(&fixture, seq, &capabilities)?;
    let decode = compile(&fixture, 1, &capabilities)?;

    let mut device = CpuDevice::new();
    let weights = upload_weights(&mut device, &fixture)?;
    build_engine(config, &fixture, device, &weights, prefill, decode)?;
    Ok(())
}

fn run_prefill(config: &Config, input_ids: &[u32]) -> Result<Vec<f32>, RuntimeError> {
    let seq = u32::try_from(input_ids.len()).map_err(|_| {
        RuntimeError::Usage("--input-ids contains more tokens than u32 can address".to_owned())
    })?;
    if seq == 0 {
        return Err(RuntimeError::Usage(
            "--input-ids must contain at least one token id".to_owned(),
        ));
    }
    if seq > config.engine.max_context {
        return Err(RuntimeError::Usage(format!(
            "sequence length {seq} exceeds engine.max_context {}",
            config.engine.max_context
        )));
    }
    if seq > MAX_SEQ {
        return Err(RuntimeError::Usage(format!(
            "sequence length {seq} exceeds the fixture limit of {MAX_SEQ}"
        )));
    }
    if let Some((index, id)) = input_ids.iter().enumerate().find(|(_, id)| **id >= VOCAB) {
        return Err(RuntimeError::Usage(format!(
            "token id {id} at position {index} is outside the fixture vocabulary 0..{}",
            VOCAB - 1
        )));
    }

    let fixture = Fixture::build();
    let capabilities = cpu_capabilities();
    let prefill = compile(&fixture, seq, &capabilities)?;
    let decode = compile(&fixture, 1, &capabilities)?;

    let mut device = CpuDevice::new();
    let weights = upload_weights(&mut device, &fixture)?;
    let engine = build_engine(config, &fixture, device, &weights, prefill, decode)?;

    step_prefill(engine, input_ids, seq)
}

fn compile(
    fixture: &Fixture,
    seq: u32,
    capabilities: &DeviceCapabilities,
) -> Result<CommandBuffer, RuntimeError> {
    fixture
        .compile(seq, capabilities)
        .map_err(|error| RuntimeError::Compile(error.to_string()))
}

fn upload_weights(
    device: &mut CpuDevice,
    fixture: &Fixture,
) -> Result<Vec<(ValueId, BufferId)>, RuntimeError> {
    let mut bindings = Vec::with_capacity(fixture.weights.len());
    for (value, data) in &fixture.weights {
        let buffer = device
            .allocate_f32(data.len(), BUFFER_ALIGNMENT)
            .map_err(device_error)?;
        device.upload_f32(buffer, data).map_err(device_error)?;
        bindings.push((*value, buffer));
    }
    Ok(bindings)
}

fn build_engine(
    config: &Config,
    fixture: &Fixture,
    device: CpuDevice,
    weights: &[(ValueId, BufferId)],
    prefill: CommandBuffer,
    decode: CommandBuffer,
) -> Result<Engine, RuntimeError> {
    let prefill = CompiledProgram {
        ir: prefill,
        weights: weights.to_vec(),
        token_ids: fixture.token_ids,
        positions: fixture.positions,
        seq_len: fixture.seq_len,
        past_len: None,
        logits: fixture.logits,
    };
    let decode = CompiledProgram {
        ir: decode,
        weights: weights.to_vec(),
        token_ids: fixture.token_ids,
        positions: fixture.positions,
        seq_len: fixture.seq_len,
        past_len: None,
        logits: fixture.logits,
    };
    let engine_config = CoreEngineConfig {
        prefill,
        decode,
        device: Box::new(device),
        max_batch: config.engine.max_batch,
        max_context: config.engine.max_context,
    };
    Engine::build(engine_config).map_err(|error| match error {
        BuildError::Device(device) => RuntimeError::Device(device.to_string()),
        other => RuntimeError::Model(other.to_string()),
    })
}

fn step_prefill(mut engine: Engine, input_ids: &[u32], seq: u32) -> Result<Vec<f32>, RuntimeError> {
    let positions: Vec<u32> = (0..seq).collect();
    let mut logits = vec![0.0_f32; VOCAB as usize];
    let input = StepInput {
        token_ids: input_ids,
        positions: &positions,
        seq_len: seq,
        past_len: 0,
    };
    engine
        .step(StepMode::Prefill, input, &mut logits)
        .map_err(|error| match error {
            StepError::Device(device) => RuntimeError::Device(device.to_string()),
            other => RuntimeError::Model(other.to_string()),
        })?;
    Ok(logits)
}

fn device_error(error: DeviceError) -> RuntimeError {
    RuntimeError::Device(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::{
        cpu_capabilities, dispatch, parse_args, parse_input_ids, serve, Command, Describe,
        RuntimeError, EXIT_COMPILE, EXIT_DEVICE, EXIT_OUTPUT, EXIT_UNIMPLEMENTED, EXIT_USAGE,
    };
    use crate::config::Config;
    use crate::config::ConfigError;
    use sl_ir::IR_VERSION;
    use std::path::PathBuf;

    fn args(items: &[&str]) -> Vec<String> {
        items.iter().map(|item| (*item).to_owned()).collect()
    }

    fn example_config() -> Config {
        Config::from_json_str(
            r#"{
                "model":  { "kind": "fixture", "architecture": "fixture" },
                "engine": { "backend": "cpu", "max_context": 16, "max_batch": 1 },
                "server": { "bind": "127.0.0.1", "port": 8000 }
            }"#,
        )
        .expect("example config")
    }

    #[test]
    fn parses_every_subcommand() {
        assert_eq!(
            parse_args(args(&["describe", "--config", "runtime.json"])),
            Ok(Command::Describe {
                config: PathBuf::from("runtime.json")
            })
        );
        assert_eq!(
            parse_args(args(&["validate", "--config=runtime.json"])),
            Ok(Command::Validate {
                config: PathBuf::from("runtime.json")
            })
        );
        assert_eq!(
            parse_args(args(&["run", "--config", "c.json", "--input-ids", "1,2"])),
            Ok(Command::Run {
                config: PathBuf::from("c.json"),
                input_ids: vec![1, 2],
            })
        );
        assert_eq!(
            parse_args(args(&["serve", "--config", "c.json"])),
            Ok(Command::Serve {
                config: PathBuf::from("c.json")
            })
        );
        assert_eq!(parse_args(args(&["version"])), Ok(Command::Version));
        assert_eq!(parse_args(args(&["--help"])), Ok(Command::Help));
    }

    #[test]
    fn input_ids_allow_spaces_but_reject_garbage() {
        assert_eq!(parse_input_ids("1, 2,0").unwrap(), vec![1, 2, 0]);
        assert_eq!(parse_input_ids("4294967295").unwrap(), vec![u32::MAX]);
        for bad in ["", "  ", "1,", "1,,2", ",1", "1,x", "-1", "4294967296"] {
            assert!(parse_input_ids(bad).is_err(), "input {bad:?} must fail");
        }
    }

    #[test]
    fn rejects_bad_command_lines() {
        let cases: &[&[&str]] = &[
            &[],
            &["bogus"],
            &["describe"],
            &["describe", "--config"],
            &["describe", "--config", "a", "--config", "b"],
            &["describe", "--config", "a", "--input-ids", "1"],
            &["describe", "", "--config", "a"],
            &["run", "--config", "a"],
            &["run", "--input-ids", "1,2"],
            &["run", "--config", "a", "--input-ids", "1,2", "extra"],
            &["version", "now"],
            &["help", "me"],
        ];
        for case in cases {
            assert!(
                parse_args(args(case)).is_err(),
                "arguments {case:?} must be rejected"
            );
        }
    }

    #[test]
    fn describe_reports_the_arch_fields() {
        let config = example_config();
        let describe = Describe::from_config(&config);
        let value = serde_json::to_value(&describe).expect("describe is serializable");
        assert_eq!(value["engine"], "starlight");
        assert_eq!(value["engine_version"], crate::VERSION);
        assert_eq!(value["runtime_version"], crate::VERSION);
        assert_eq!(value["ir_version"], IR_VERSION);
        assert_eq!(value["backend"], "cpu");
        assert_eq!(value["architectures"][0], "fixture");
        assert_eq!(value["max_context"], 16);
        assert_eq!(value["max_batch"], 1);
        assert_eq!(value["capabilities"][0], "token-ids");
        assert_eq!(describe.engine, "starlight");
    }

    #[test]
    fn cpu_capabilities_cover_the_fixture() {
        let capabilities = cpu_capabilities();
        assert_eq!(capabilities.validate(), Ok(()));
        assert!(capabilities.supports(sl_ir::OpCode::Gather));
        assert!(capabilities.supports(sl_ir::OpCode::RmsNorm));
        assert!(capabilities.supports(sl_ir::OpCode::MatMulNT));
        assert!(capabilities.supports(sl_ir::OpCode::LastToken));
        assert_eq!(capabilities.min_buffer_align, 64);
        assert_eq!(capabilities.backend.0, "cpu");
        assert_eq!(super::BUFFER_ALIGNMENT, 64);
    }

    #[test]
    fn version_and_help_dispatch() {
        let mut output = Vec::new();
        dispatch(&Command::Version, &mut output).expect("version");
        let text = String::from_utf8(output).expect("utf8");
        assert!(text.contains(crate::VERSION));
        assert!(text.contains(&format!("ir_version {IR_VERSION}")));

        let mut output = Vec::new();
        dispatch(&Command::Help, &mut output).expect("help");
        let text = String::from_utf8(output).expect("utf8");
        assert!(text.contains("starlight-runtime run --config"));
    }

    #[test]
    fn serve_is_explicitly_unsupported() {
        let error = serve(&example_config()).expect_err("serve must be unsupported");
        assert!(matches!(error, RuntimeError::Unsupported(_)));
        assert!(error.to_string().contains("sl-daemon"));
        assert_eq!(error.exit_code(), EXIT_UNIMPLEMENTED);
    }

    #[test]
    fn dispatch_reports_missing_config_before_serve() {
        let path = std::env::temp_dir().join(format!(
            "starlight-runtime-missing-serve-{}.json",
            std::process::id()
        ));
        let mut output = Vec::new();
        let error = dispatch(&Command::Serve { config: path }, &mut output)
            .expect_err("missing config must fail");
        assert!(matches!(
            error,
            RuntimeError::Config(ConfigError::Io { .. })
        ));
        assert_eq!(error.exit_code(), EXIT_USAGE);
    }

    #[test]
    fn dispatch_rejects_sequence_over_the_config_limit() {
        let path = std::env::temp_dir().join(format!(
            "starlight-runtime-small-context-{}.json",
            std::process::id()
        ));
        std::fs::write(
            &path,
            r#"{"model":{"kind":"fixture","architecture":"fixture"},"engine":{"backend":"cpu","max_context":1,"max_batch":1},"server":{"bind":"127.0.0.1","port":8000}}"#,
        )
        .expect("write temp config");
        let mut output = Vec::new();
        let result = dispatch(
            &Command::Run {
                config: path.clone(),
                input_ids: vec![1, 2],
            },
            &mut output,
        );
        let _ = std::fs::remove_file(&path);
        let error = result.expect_err("sequence above max_context must fail");
        assert!(matches!(error, RuntimeError::Usage(_)));
        assert_eq!(error.exit_code(), EXIT_USAGE);
    }

    #[test]
    fn dispatch_rejects_token_ids_outside_the_vocabulary() {
        let path = std::env::temp_dir().join(format!(
            "starlight-runtime-bad-id-{}.json",
            std::process::id()
        ));
        std::fs::write(
            &path,
            r#"{"model":{"kind":"fixture","architecture":"fixture"},"engine":{"backend":"cpu","max_context":16,"max_batch":1},"server":{"bind":"127.0.0.1","port":8000}}"#,
        )
        .expect("write temp config");
        let mut output = Vec::new();
        let result = dispatch(
            &Command::Run {
                config: path.clone(),
                input_ids: vec![1, 8],
            },
            &mut output,
        );
        let _ = std::fs::remove_file(&path);
        let error = result.expect_err("id outside the vocabulary must fail");
        assert!(matches!(error, RuntimeError::Usage(_)));
        assert!(error.to_string().contains("vocabulary"));
    }

    #[test]
    fn dispatch_rejects_sequences_over_the_fixture_limit() {
        let path = std::env::temp_dir().join(format!(
            "starlight-runtime-long-seq-{}.json",
            std::process::id()
        ));
        std::fs::write(
            &path,
            r#"{"model":{"kind":"fixture","architecture":"fixture"},"engine":{"backend":"cpu","max_context":64,"max_batch":1},"server":{"bind":"127.0.0.1","port":8000}}"#,
        )
        .expect("write temp config");
        let mut output = Vec::new();
        let result = dispatch(
            &Command::Run {
                config: path.clone(),
                input_ids: vec![1; 9],
            },
            &mut output,
        );
        let _ = std::fs::remove_file(&path);
        let error = result.expect_err("sequence above the fixture limit must fail");
        assert!(error.to_string().contains("fixture limit"));
        assert_eq!(error.exit_code(), EXIT_USAGE);
    }

    #[test]
    fn error_exit_codes_are_stable() {
        let cases = [
            (RuntimeError::Usage("bad".to_owned()), EXIT_USAGE),
            (RuntimeError::Compile("bad".to_owned()), EXIT_COMPILE),
            (RuntimeError::Model("bad".to_owned()), EXIT_COMPILE),
            (RuntimeError::Device("bad".to_owned()), EXIT_DEVICE),
            (
                RuntimeError::Unsupported("bad".to_owned()),
                EXIT_UNIMPLEMENTED,
            ),
            (RuntimeError::Output("bad".to_owned()), EXIT_OUTPUT),
        ];
        for (error, expected) in cases {
            assert_eq!(error.exit_code(), expected, "{error}");
        }
        let config_error = RuntimeError::Config(ConfigError::Validation("bad".to_owned()));
        assert_eq!(config_error.exit_code(), EXIT_USAGE);
        assert!(std::error::Error::source(&config_error).is_some());
    }
}
