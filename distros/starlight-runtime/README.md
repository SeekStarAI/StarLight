# starlight-runtime

`starlight-runtime` is the **composition root** of the StarLight inference
stack (ARCH-0005 §5.5, ARCH-0007 §3.4). It is an executable, not a library of
kernels: it parses configuration, compiles the deterministic tiny fixture with
`sl-graph`, uploads weights through a `CpuDevice`, drives `sl_core::Engine` for
one prefill step and prints logits.

The real pipeline exercised by `run` is:

```text
config.json
  → sl_graph::compile(Program, ShapeSpec, DeviceCapabilities)      (prefill + decode)
  → sl_device::Device / starlight_backend_cpu::CpuDevice::new()
  → upload Param buffers
  → sl_core::Engine::build(EngineConfig)
  → Engine::step(Prefill, StepInput) → logits
```

## What it is

- The assembler of `sl-graph`, `sl-core`, `sl-device` and `backend-cpu`.
- The owner of the process contract from ARCH-0005 §5: `describe`,
  `validate`, `run`, `serve`, `version`, JSON configuration and exit codes.
- A consumer of **token IDs**, not text. Until `sl-tokenizer` exists the input
  is `--input-ids`.
- A tiny, deterministic fixture runner for the ARCH-0007 end-to-end acceptance.

## What it is not

- Not a kernel or operator implementation (that is `backend-cpu`).
- Not a graph optimizer or memory planner (that is `sl-graph`).
- Not an engine, scheduler or session manager (that is `sl-core`).
- Not an HTTP server, OpenAI implementation, tokenizer, sampler or queue (those
  belong to `sl-daemon` / `sl-tokenizer`).
- Not a model loader: `model.kind` is currently limited to `fixture`.
- Not a multi-model or multi-node runtime (ARCH-0005 v1: one process = one
  engine).

## CLI

```text
starlight-runtime describe --config <path>
starlight-runtime validate --config <path>
starlight-runtime run --config <path> --input-ids 1,2
starlight-runtime serve --config <path>
starlight-runtime version
starlight-runtime --help
```

Flags accept both `--config path` and `--config=path`. `run` additionally
requires a comma-separated `u32` list without empty elements.

| Command | Behavior |
| --- | --- |
| `describe` | Prints the ARCH-0005 §5.2 JSON. Does not load weights. |
| `validate` | Loads the fixture, compiles prefill (`seq = min(max_context, 2)`) and decode (`seq = 1`), uploads weights, builds the engine and prints `ok`. It does not call `step`. |
| `run` | Compiles prefill at `len(input_ids)` and decode at `1`, builds the engine, runs one `Prefill` step and prints one logit per line. |
| `serve` | Loads the configuration, then fails with an explicit `Unsupported` error (exit 5). No HTTP is started. |
| `version` | Prints the crate version and `IR_VERSION`. |
| `--help` | Prints usage. |

Example:

```sh
cat > /tmp/runtime.json <<'JSON'
{
  "model":  { "kind": "fixture", "architecture": "fixture" },
  "engine": { "backend": "cpu", "max_context": 16, "max_batch": 1 },
  "server": { "bind": "127.0.0.1", "port": 8000 }
}
JSON

cargo run -p starlight-runtime -- describe --config /tmp/runtime.json
cargo run -p starlight-runtime -- run --config /tmp/runtime.json --input-ids 1,2
```

`describe` reports:

```json
{
  "engine": "starlight",
  "engine_version": "0.1.0",
  "ir_version": 2,
  "runtime_version": "0.1.0",
  "backend": "cpu",
  "architectures": ["fixture"],
  "max_context": 16,
  "max_batch": 1,
  "quantization": "none",
  "capabilities": ["token-ids"]
}
```

`capabilities` is deliberately honest: the runtime accepts token IDs and
returns logits, and it does not implement text, streaming or serving.

## Configuration

Configuration is a single JSON document (ARCH-0007 §3.4; JSON avoids a TOML
dependency). `--config` is the only input; no environment variable changes its
meaning.

```json
{
  "model":  { "kind": "fixture", "architecture": "fixture" },
  "engine": { "backend": "cpu", "max_context": 16, "max_batch": 1 },
  "server": { "bind": "127.0.0.1", "port": 8000 }
}
```

Validation rules:

- `engine.max_context > 0` and `engine.max_batch > 0`;
- `engine.backend` must be known (`cpu` today); unknown backends are rejected,
  not ignored;
- `model.kind` must be known (`fixture` today);
- `model.architecture` and `server.bind` must be non-empty;
- unknown JSON fields are ignored by `serde`.

`server` is parsed and reported, but nothing binds to it until `serve` is
implemented.

Note: the example keeps `max_context = 16` as in ARCH-0007, while the fixture's
symbolic sequence bound is 8. `run` therefore rejects requests longer than 8
even when `max_context` is larger.

## Fixture

`src/fixture.rs` builds a deterministic graph, without files, randomness or
`sl-model-adapters`:

```text
token_ids ─ Gather(embed, token_ids) ─ RmsNorm(x, norm_weight)
          ─ MatMulNT(rms, lm_head) ─ LastToken(matmul, seq_len) ─ logits
```

Values:

| Value | Role | Type | Shape |
| --- | --- | --- | --- |
| `token_ids` | Input | U32 | `[1, Sym(seq)]` |
| `embed` | Param | F32 | `[8, 4]` |
| `gather` | Temp | F32 | `[1, Sym(seq), 4]` |
| `norm_weight` | Param | F32 | `[4]` |
| `rms` | Temp | F32 | `[1, Sym(seq), 4]` |
| `lm_head` | Param | F32 | `[8, 4]` |
| `matmul` | Temp | F32 | `[1, Sym(seq), 8]` |
| `seq_len` | Scalar | U32 | `[]` |
| `logits` | Output | F32 | `[1, 8]` |

Nodes: `Gather`, `RmsNorm { eps: 1e-5 }`, `MatMulNT` (2-D weight on the right,
prefix-broadcast 3-D activation) and `LastToken`.

Weights are constants:

- `embed[i] = (i as f32 - 3.0) * 0.1` for `i in 0..32`;
- `norm_weight = [1.0, 0.5, 1.5, 0.75]`;
- `lm_head[i] = ((i % 5) as f32 - 2.0) * 0.1` for `i in 0..32`.

The same `Program` is compiled twice with different `ShapeSpec`s: prefill uses
`seq = len(input_ids)` (2 in the CLI example) and decode uses `seq = 1`. The
fixture has no RoPE, so `positions` is `None`; the runtime still passes
`0..seq_len` as `StepInput::positions`.

The end-to-end test compares the real engine output against an independent
`f64` implementation of gather → RMSNorm → matmul → last token, with tolerance
`1e-5`.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | success |
| 1 | unexpected output I/O failure |
| 2 | configuration or command-line error (missing config, bad JSON, invalid ids, sequence over the limit) |
| 3 | graph compilation or engine build error |
| 4 | device error (allocation, upload, step) |
| 5 | not implemented (`serve`) |

Errors are printed to stderr as `error: ...`.

## API dependency

The runtime is written against the frozen interfaces in ARCH-0007 §3 and adds
no stubs or mocks:

| API | Used for |
| --- | --- |
| `sl_ir::{Program, ShapeSpec, DeviceCapabilities, IR_VERSION}` | fixture, compile input and version reporting |
| `sl_graph::compile` / `sl_graph::CompileError` | prefill and decode compilation |
| `sl_device::{Device, DeviceError}` | weight allocation/upload and device errors |
| `starlight_backend_cpu::CpuDevice::new()` | CPU device |
| `sl_core::{Engine, EngineConfig, CompiledProgram, StepInput, StepMode}` | engine build and one prefill step |

ARCH-0007 freezes `compile`, `Engine` and `Device`, but not capability
discovery. Because there is no `CpuDevice::capabilities()` in the contract,
`runtime::cpu_capabilities()` declares exactly what the fixture needs: the four
core operators, `F32`/`U32` and 64-byte alignment. If the backend later exposes
its own capabilities, this function should delegate to it.

## `serve` is not implemented

`sl-daemon`, `sl-tokenizer` and the HTTP surface do not exist yet. Per
ARCH-0005 §4 and ARCH-0007 §3.4, the runtime must not grow a private HTTP
server, so `serve` always returns:

```text
error: unsupported: serve is not implemented: sl-daemon is not ready (requested bind 127.0.0.1:8000)
```

with exit code 5. The `server` configuration section is parsed and included in
the error, but nothing listens on the port.

## Tests

```sh
cargo test -p starlight-runtime
cargo clippy -p starlight-runtime --all-targets -- -D warnings
cargo fmt --all --check
```

Coverage (43 tests):

- `src/config.rs`: parsing, missing/malformed JSON, zero limits, unknown
  backend/kind, I/O errors.
- `src/runtime.rs`: CLI parsing, `describe` fields, capability declaration,
  version/help, `serve` being unsupported, sequence/token-id limits, exit-code
  mapping.
- `tests/fixture_compile.rs`: two compilations (`seq = 2`, `seq = 1`),
  `CommandBuffer::validate(64)`, deterministic output, bound rejection.
- `tests/end_to_end.rs`: real `compile → CpuDevice → Engine::build →
  step(Prefill)` chain versus the `f64` oracle, plus `dispatch(run)` and
  `dispatch(validate)`.
- `tests/cli.rs`: black-box exit codes, `describe` JSON, bad input handling.

## Current status

The real `sl-graph`, `sl-core`, `sl-device` and
`starlight_backend_cpu::CpuDevice` implementations have landed, and the full
end-to-end chain passes (43 tests). No mocks or stubs are used.

Workspace-level checks pass:

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
```
