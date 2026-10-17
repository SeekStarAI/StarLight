// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 SeekStarAI and StarLight Contributors

//! Runtime configuration.
//!
//! The runtime consumes a single JSON document (ARCH-0007 §3.4):
//!
//! ```json
//! {
//!   "model":  { "kind": "fixture", "architecture": "fixture" },
//!   "engine": { "backend": "cpu", "max_context": 16, "max_batch": 1 },
//!   "server": { "bind": "127.0.0.1", "port": 8000 }
//! }
//! ```
//!
//! Parsing and validation are separate steps: [`Config::from_json_str`] only
//! deserializes, [`Config::validate`] enforces the runtime's invariants, and
//! [`Config::load`] does both.

use std::fmt;
use std::fs;
use std::path::Path;

use serde::Deserialize;

/// Backends the runtime can build an engine for today.
pub const SUPPORTED_BACKENDS: &[&str] = &["cpu"];

/// Model kinds the runtime can load today.
pub const SUPPORTED_MODEL_KINDS: &[&str] = &["fixture"];

/// Top-level runtime configuration.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize)]
pub struct Config {
    /// Model source and architecture metadata.
    pub model: ModelConfig,
    /// Engine limits and backend selection.
    pub engine: EngineConfig,
    /// Process-level server settings. Parsed now, used when `serve` lands.
    pub server: ServerConfig,
}

/// `model` section.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize)]
pub struct ModelConfig {
    /// Model source kind; only `fixture` is understood.
    pub kind: String,
    /// Model architecture reported by `describe`.
    pub architecture: String,
}

/// `engine` section.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize)]
pub struct EngineConfig {
    /// Backend name; only `cpu` is understood.
    pub backend: String,
    /// Maximum context length in tokens.
    pub max_context: u32,
    /// Maximum batch size.
    pub max_batch: u32,
}

/// `server` section.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize)]
pub struct ServerConfig {
    /// Bind address; recorded for the future `serve` implementation.
    pub bind: String,
    /// Bind port; recorded for the future `serve` implementation.
    pub port: u16,
}

impl Config {
    /// Loads and validates a configuration file.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::Io`] when the file cannot be read,
    /// [`ConfigError::Parse`] when it is not valid JSON for this schema, and
    /// [`ConfigError::Validation`] when the values are not usable.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let text = fs::read_to_string(path).map_err(|source| ConfigError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let config: Self = serde_json::from_str(&text).map_err(ConfigError::Parse)?;
        config.validate()?;
        Ok(config)
    }

    /// Parses configuration from a JSON string without validating it.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::Parse`] when the document does not match the
    /// schema.
    pub fn from_json_str(text: &str) -> Result<Self, ConfigError> {
        serde_json::from_str(text).map_err(ConfigError::Parse)
    }

    /// Validates the parsed configuration.
    ///
    /// Unknown backends and model kinds are rejected rather than silently
    /// ignored, because neither `run` nor `validate` can honor them.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::Validation`] describing the first violated
    /// invariant.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.engine.max_context == 0 {
            return Err(ConfigError::Validation(
                "engine.max_context must be greater than zero".to_owned(),
            ));
        }
        if self.engine.max_batch == 0 {
            return Err(ConfigError::Validation(
                "engine.max_batch must be greater than zero".to_owned(),
            ));
        }
        if !SUPPORTED_BACKENDS.contains(&self.engine.backend.as_str()) {
            return Err(ConfigError::Validation(format!(
                "unsupported engine.backend {:?}; supported backends: {}",
                self.engine.backend,
                SUPPORTED_BACKENDS.join(", ")
            )));
        }
        if !SUPPORTED_MODEL_KINDS.contains(&self.model.kind.as_str()) {
            return Err(ConfigError::Validation(format!(
                "unsupported model.kind {:?}; supported kinds: {}",
                self.model.kind,
                SUPPORTED_MODEL_KINDS.join(", ")
            )));
        }
        if self.model.architecture.trim().is_empty() {
            return Err(ConfigError::Validation(
                "model.architecture must not be empty".to_owned(),
            ));
        }
        if self.server.bind.trim().is_empty() {
            return Err(ConfigError::Validation(
                "server.bind must not be empty".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Failure while reading, parsing or validating a [`Config`].
#[derive(Debug)]
pub enum ConfigError {
    /// The configuration file could not be read.
    Io {
        /// Path that was being read.
        path: std::path::PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// The configuration document is not valid JSON for this schema.
    Parse(serde_json::Error),
    /// The configuration parsed but violates a runtime invariant.
    Validation(String),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(
                    formatter,
                    "failed to read config {}: {source}",
                    path.display()
                )
            }
            Self::Parse(source) => write!(formatter, "failed to parse config: {source}"),
            Self::Validation(message) => write!(formatter, "invalid config: {message}"),
        }
    }
}

impl std::error::Error for ConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Parse(source) => Some(source),
            Self::Validation(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Config, ConfigError};

    const EXAMPLE: &str = r#"{
        "model":  { "kind": "fixture", "architecture": "fixture" },
        "engine": { "backend": "cpu", "max_context": 16, "max_batch": 1 },
        "server": { "bind": "127.0.0.1", "port": 8000 }
    }"#;

    fn example() -> Config {
        Config::from_json_str(EXAMPLE).expect("example config must parse")
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "starlight-runtime-config-{}-{name}",
            std::process::id()
        ))
    }

    #[test]
    fn parses_the_arch_example() {
        let config = example();
        assert_eq!(config.model.kind, "fixture");
        assert_eq!(config.model.architecture, "fixture");
        assert_eq!(config.engine.backend, "cpu");
        assert_eq!(config.engine.max_context, 16);
        assert_eq!(config.engine.max_batch, 1);
        assert_eq!(config.server.bind, "127.0.0.1");
        assert_eq!(config.server.port, 8000);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn rejects_missing_sections() {
        let error =
            Config::from_json_str(r#"{"model":{"kind":"fixture","architecture":"fixture"}}"#)
                .expect_err("missing sections must fail");
        assert!(matches!(error, ConfigError::Parse(_)));
    }

    #[test]
    fn rejects_malformed_json() {
        let error = Config::from_json_str("{not json").expect_err("malformed JSON must fail");
        assert!(matches!(error, ConfigError::Parse(_)));
    }

    #[test]
    fn rejects_zero_max_context() {
        let mut config = example();
        config.engine.max_context = 0;
        let error = config.validate().expect_err("zero max_context");
        assert!(error.to_string().contains("max_context"));
    }

    #[test]
    fn rejects_zero_max_batch() {
        let mut config = example();
        config.engine.max_batch = 0;
        let error = config.validate().expect_err("zero max_batch");
        assert!(error.to_string().contains("max_batch"));
    }

    #[test]
    fn rejects_unknown_backend() {
        let mut config = example();
        config.engine.backend = "vulkan".to_owned();
        let error = config.validate().expect_err("unknown backend");
        assert!(error.to_string().contains("unsupported engine.backend"));
    }

    #[test]
    fn rejects_unknown_model_kind() {
        let mut config = example();
        config.model.kind = "safetensors".to_owned();
        let error = config.validate().expect_err("unknown model kind");
        assert!(error.to_string().contains("unsupported model.kind"));
    }

    #[test]
    fn rejects_empty_architecture_and_bind() {
        let mut config = example();
        config.model.architecture = "  ".to_owned();
        assert!(config.validate().is_err());

        let mut config = example();
        config.server.bind.clear();
        assert!(config.validate().is_err());
    }

    #[test]
    fn load_reports_missing_file_as_io() {
        let path = temp_path("missing.json");
        let error = Config::load(&path).expect_err("missing file must fail");
        assert!(matches!(error, ConfigError::Io { .. }));
        assert!(std::error::Error::source(&error).is_some());
    }

    #[test]
    fn load_rejects_invalid_file_contents() {
        let path = temp_path("invalid.json");
        std::fs::write(
            &path,
            r#"{"model":{"kind":"other","architecture":"x"},"engine":{"backend":"cpu","max_context":1,"max_batch":1},"server":{"bind":"127.0.0.1","port":1}}"#,
        )
        .expect("write temp config");
        let error = Config::load(&path).expect_err("invalid kind must fail");
        let _ = std::fs::remove_file(&path);
        assert!(matches!(error, ConfigError::Validation(_)));
    }
}
