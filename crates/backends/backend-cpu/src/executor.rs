// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 SeekStarAI and StarLight Contributors

//! Reference executor for [`sl_ir::CommandBuffer`].
//!
//! This module is a **reference executor**, not a device backend: it runs the
//! command stream on the host, in command order, using the scalar kernels in
//! [`crate::reference`]. It exists to exercise the IR and the kernels together
//! and to serve as the semantic baseline for real backends. It performs no
//! scheduling, fusion, queueing or asynchronous completion.
//!
//! # Stores
//!
//! The caller fills three build-time stores before calling [`execute`]:
//!
//! - [`BindingTable`] maps every non-`Temp`, non-`Scalar` value to a
//!   [`BufferId`].
//! - [`BufferStore`] maps a [`BufferId`] to `f32` data (and optionally exact
//!   `u32` data for integer tensors such as `positions` and `ids`).
//! - [`ScalarStore`] maps a `Scalar` value to its per-step `u32` value.
//!
//! `Temp` values do not have individual buffers: they are slices of one shared
//! scratchpad buffer, bound to [`SCRATCHPAD`] (`BufferId(0)`). The slice of a
//! `Temp` is described by [`sl_ir::MemoryPlan`], whose offsets and sizes are
//! **byte** quantities. The executor validates that every offset and size is a
//! multiple of four and that a slot's size equals the value's byte size.
//!
//! # Execution order and borrows
//!
//! Commands run strictly in `ir.commands` order; `deps` is ignored because v1
//! command buffers are a total order. A command's input buffers are taken out
//! of [`BufferStore`], the command runs, and the buffers are put back. The
//! scratchpad is taken once per command and split into disjoint `Temp` slices
//! with `split_at_mut`, so the executor needs neither `unsafe` nor heap
//! allocation on the execution path. Buffers are restored even when a command
//! returns an error.
//!
//! # Integer tensors
//!
//! The reference kernels take `&[u32]` for embedding ids and RoPE positions.
//! Callers should insert such buffers with [`BufferStore::insert_u32`]. For
//! convenience, a `u32` value may also be backed by an `f32` buffer whose
//! elements are non-negative integers; the executor converts it with a
//! conversion vector that is allocated during the build phase and reused
//! during execution. At most one such converted tensor is needed by any v1
//! operator.

use core::fmt;

use sl_ir::{
    BufferId, Command, CommandBuffer, DType, Effect, Op, OpCode, PortSpec, StateKind, Value,
    ValueId, ValueRole,
};

use crate::reference::{self, KernelError};

/// Buffer id reserved for the shared scratchpad.
///
/// Every `Temp` value is a byte range of the buffer bound to this id. A
/// command port must not bind to `SCRATCHPAD` under any other role.
pub const SCRATCHPAD: BufferId = BufferId(0);

/// Maximum number of dimensions the reference executor inspects.
///
/// The v1 operator signatures allow at most four dimensions; the larger limit
/// leaves room for future prefix dimensions without changing the data model.
const MAX_RANK: usize = 8;

/// Maximum number of input plus output ports of one command.
///
/// The largest v1 signature has four inputs and one output.
const MAX_PORTS: usize = 6;

/// Largest integer payload the build-time conversion scratch vector reserves.
///
/// `insert_u32` is the exact and unbounded path; the `f32`-backed convenience
/// path only covers realistic token-id and position tensors without reserving
/// memory proportional to large weight buffers.
const CONVERSION_CAPACITY_LIMIT: usize = 1 << 20;

/// Error returned by [`execute`].
///
/// Every variant is allocation-free. Kernel-level failures are wrapped in
/// [`ExecutionError::Kernel`] instead of being flattened, so callers can still
/// distinguish a length mismatch from an invalid shape.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionError {
    /// The command uses an operator the reference executor does not implement.
    UnsupportedOp(OpCode),
    /// A value that needs a buffer has no entry in the [`BindingTable`].
    MissingBinding(ValueId),
    /// A binding refers to a buffer that is absent from the [`BufferStore`].
    MissingBuffer(BufferId),
    /// A `Scalar` value has no entry in the [`ScalarStore`].
    MissingScalar(ValueId),
    /// The command buffer's memory plan violates an executor invariant.
    ///
    /// The payload is a static description such as
    /// `"slot size does not match the value size"`.
    InvalidPlan(&'static str),
    /// A reference kernel rejected its inputs.
    Kernel(KernelError),
}

impl fmt::Display for ExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedOp(op) => write!(formatter, "unsupported operator {op:?}"),
            Self::MissingBinding(value) => {
                write!(formatter, "no buffer binding for value {}", value.0)
            }
            Self::MissingBuffer(buffer) => {
                write!(
                    formatter,
                    "buffer {} is not present in the buffer store",
                    buffer.0
                )
            }
            Self::MissingScalar(value) => {
                write!(formatter, "no scalar value for value {}", value.0)
            }
            Self::InvalidPlan(reason) => write!(formatter, "invalid memory plan: {reason}"),
            Self::Kernel(error) => write!(formatter, "kernel error: {error}"),
        }
    }
}

impl core::error::Error for ExecutionError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Kernel(error) => Some(error),
            _ => None,
        }
    }
}

/// Build-time mapping from a value to the buffer that backs it.
///
/// Only `Param`, `Input`, `Output` and `State` values use this table; `Temp`
/// values live in the scratchpad and `Scalar` values live in a
/// [`ScalarStore`]. Inserting the same value twice replaces the previous
/// binding.
#[derive(Clone, Debug, Default)]
pub struct BindingTable {
    entries: Vec<(ValueId, BufferId)>,
}

impl BindingTable {
    /// Creates an empty binding table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Binds `value` to `buffer`, replacing an existing binding for `value`.
    pub fn insert(&mut self, value: ValueId, buffer: BufferId) {
        if let Some(entry) = self.entries.iter_mut().find(|(id, _)| *id == value) {
            entry.1 = buffer;
        } else {
            self.entries.push((value, buffer));
        }
    }

    /// Returns the buffer bound to `value`.
    pub fn get(&self, value: ValueId) -> Option<BufferId> {
        self.entries
            .iter()
            .find(|(id, _)| *id == value)
            .map(|(_, buffer)| *buffer)
    }
}

/// Build-time mapping from a `Scalar` value to its per-step `u32` value.
///
/// Scalar values are `u32` and are not stored in the [`BufferStore`]. Inserting
/// the same value twice replaces the previous scalar.
#[derive(Clone, Debug, Default)]
pub struct ScalarStore {
    entries: Vec<(ValueId, u32)>,
}

impl ScalarStore {
    /// Creates an empty scalar store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the scalar value bound to `value`.
    pub fn insert(&mut self, value: ValueId, scalar: u32) {
        if let Some(entry) = self.entries.iter_mut().find(|(id, _)| *id == value) {
            entry.1 = scalar;
        } else {
            self.entries.push((value, scalar));
        }
    }

    /// Returns the scalar bound to `value`.
    pub fn get(&self, value: ValueId) -> Option<u32> {
        self.entries
            .iter()
            .find(|(id, _)| *id == value)
            .map(|(_, scalar)| *scalar)
    }
}

/// Caller-owned storage for `f32` and `u32` buffers.
///
/// Callers insert buffers during the build phase. [`execute`] takes the
/// buffers it needs out of the store, runs the command, and puts them back, so
/// no `Vec` is allocated on the execution path. A missing buffer is reported
/// as [`ExecutionError::MissingBuffer`].
#[derive(Clone, Debug, Default)]
pub struct BufferStore {
    floats: Vec<(BufferId, Option<Vec<f32>>)>,
    integers: Vec<(BufferId, Option<Vec<u32>>)>,
    /// Reused by the `u32`-as-`f32` convenience path. It is sized during
    /// [`BufferStore::insert`], never during execution.
    conversion: Vec<u32>,
}

impl BufferStore {
    /// Creates an empty buffer store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts or replaces an `f32` buffer.
    ///
    /// The conversion scratch vector is grown here so that converting an
    /// `f32` payload to `u32` later does not allocate.
    pub fn insert(&mut self, id: BufferId, data: Vec<f32>) {
        if data.len() <= CONVERSION_CAPACITY_LIMIT && data.len() > self.conversion.capacity() {
            self.conversion.reserve(data.len() - self.conversion.len());
        }
        if let Some((_, slot)) = self.floats.iter_mut().find(|(key, _)| *key == id) {
            *slot = Some(data);
        } else {
            self.floats.push((id, Some(data)));
        }
    }

    /// Inserts or replaces an exact `u32` buffer.
    ///
    /// Use this for `U32` tensors such as RoPE positions and embedding ids.
    pub fn insert_u32(&mut self, id: BufferId, data: Vec<u32>) {
        if let Some((_, slot)) = self.integers.iter_mut().find(|(key, _)| *key == id) {
            *slot = Some(data);
        } else {
            self.integers.push((id, Some(data)));
        }
    }

    /// Returns the `f32` buffer bound to `id`.
    pub fn get(&self, id: BufferId) -> Option<&[f32]> {
        self.floats
            .iter()
            .find(|(key, _)| *key == id)
            .and_then(|(_, data)| data.as_deref())
    }

    /// Returns the exact `u32` buffer bound to `id`.
    pub fn get_u32(&self, id: BufferId) -> Option<&[u32]> {
        self.integers
            .iter()
            .find(|(key, _)| *key == id)
            .and_then(|(_, data)| data.as_deref())
    }

    fn take_f32(&mut self, id: BufferId) -> Option<Vec<f32>> {
        self.floats
            .iter_mut()
            .find(|(key, _)| *key == id)
            .and_then(|(_, data)| data.take())
    }

    fn put_f32(&mut self, id: BufferId, data: Vec<f32>) {
        if let Some((_, slot)) = self.floats.iter_mut().find(|(key, _)| *key == id) {
            *slot = Some(data);
        }
    }

    fn take_u32(&mut self, id: BufferId) -> Option<Vec<u32>> {
        self.integers
            .iter_mut()
            .find(|(key, _)| *key == id)
            .and_then(|(_, data)| data.take())
    }

    fn put_u32(&mut self, id: BufferId, data: Vec<u32>) {
        if let Some((_, slot)) = self.integers.iter_mut().find(|(key, _)| *key == id) {
            *slot = Some(data);
        }
    }

    fn take_conversion(&mut self) -> Vec<u32> {
        core::mem::take(&mut self.conversion)
    }

    fn put_conversion(&mut self, data: Vec<u32>) {
        self.conversion = data;
    }
}

/// Executes every command of `ir` in order.
///
/// `bindings`, `buffers` and `scalars` must be populated during the build
/// phase. On success, all output buffers and persistent state buffers contain
/// the command stream's results and every input buffer is still present in
/// `buffers`. On error, buffers that were taken out of `buffers` are restored
/// before returning where possible.
///
/// The executor validates the memory plan and every command's ports before
/// calling a kernel. It ignores `Command::deps`; v1 command buffers are a total
/// order and are executed exactly as listed.
///
/// # Errors
///
/// Returns [`ExecutionError::MissingBinding`], [`ExecutionError::MissingBuffer`]
/// or [`ExecutionError::MissingScalar`] for incomplete bindings,
/// [`ExecutionError::InvalidPlan`] for inconsistent scratchpad layouts or port
/// aliasing, and [`ExecutionError::Kernel`] when a reference kernel rejects the
/// operand shapes or lengths.
pub fn execute(
    ir: &CommandBuffer,
    bindings: &BindingTable,
    buffers: &mut BufferStore,
    scalars: &ScalarStore,
) -> Result<(), ExecutionError> {
    validate_plan(ir, buffers)?;
    for command in &ir.commands {
        execute_command(ir, bindings, buffers, scalars, command)?;
    }
    Ok(())
}

/// A resolved shape: rank plus concrete extents.
#[derive(Clone, Copy, Debug, Default)]
struct Dims {
    rank: usize,
    extents: [usize; MAX_RANK],
}

impl Dims {
    fn from_value(value: &Value) -> Result<Self, ExecutionError> {
        let dimensions = value.shape.as_slice();
        if dimensions.len() > MAX_RANK {
            return Err(invalid_shape(
                "value rank exceeds the reference executor limit",
            ));
        }
        let mut extents = [0_usize; MAX_RANK];
        for (index, dimension) in dimensions.iter().enumerate() {
            let extent = dimension.as_const().ok_or(ExecutionError::InvalidPlan(
                "value shape still contains a symbolic dimension",
            ))?;
            extents[index] = usize::try_from(extent)
                .map_err(|_| ExecutionError::Kernel(KernelError::ShapeOverflow))?;
        }
        Ok(Self {
            rank: dimensions.len(),
            extents,
        })
    }

    fn dim(&self, index: usize) -> Result<usize, ExecutionError> {
        if index < self.rank {
            Ok(self.extents[index])
        } else {
            Err(invalid_shape("operator rank is too small"))
        }
    }

    fn last(&self) -> Result<usize, ExecutionError> {
        if self.rank == 0 {
            Err(invalid_shape("operator requires at least one dimension"))
        } else {
            Ok(self.extents[self.rank - 1])
        }
    }

    fn product(&self, end: usize) -> Result<usize, ExecutionError> {
        if end > self.rank {
            return Err(invalid_shape("operator rank is too small"));
        }
        self.extents[..end]
            .iter()
            .try_fold(1_usize, |product, &extent| {
                product
                    .checked_mul(extent)
                    .ok_or(ExecutionError::Kernel(KernelError::ShapeOverflow))
            })
    }

    fn element_count(&self) -> Result<usize, ExecutionError> {
        self.product(self.rank)
    }

    fn same_as(&self, other: &Self) -> bool {
        self.rank == other.rank && self.extents[..self.rank] == other.extents[..other.rank]
    }
}

/// Where one command port gets its bytes.
#[derive(Clone, Copy, Debug)]
enum Source {
    /// An `f32` buffer in the [`BufferStore`].
    ExternalF32(BufferId),
    /// An exact `u32` buffer in the [`BufferStore`].
    ExternalU32(BufferId),
    /// An `f32` buffer whose integer payload is converted to `u32`.
    ExternalU32FromF32(BufferId),
    /// A byte range of the shared scratchpad, in `f32` elements.
    Scratch { start: usize, len: usize },
    /// A value in the [`ScalarStore`].
    Scalar,
}

impl Source {
    fn buffer_id(self) -> Option<BufferId> {
        match self {
            Self::ExternalF32(id) | Self::ExternalU32(id) | Self::ExternalU32FromF32(id) => {
                Some(id)
            }
            Self::Scratch { .. } | Self::Scalar => None,
        }
    }
}

/// Resolved metadata for one command port.
#[derive(Clone, Copy, Debug)]
struct PortView {
    value: ValueId,
    role: ValueRole,
    dims: Dims,
    source: Source,
}

/// Resolved `KvCache` geometry needed by `KvAppend` and `CausalAttention`.
#[derive(Clone, Copy, Debug)]
struct StateLayout {
    buffer: BufferId,
    batch: usize,
    layers: usize,
    kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
}

/// All buffers and metadata of one command, taken before it runs.
struct Staged {
    ports: [PortView; MAX_PORTS],
    port_count: usize,
    input_count: usize,
    f32_values: [Option<Vec<f32>>; MAX_PORTS],
    u32_values: [Option<Vec<u32>>; MAX_PORTS],
    scalars: [Option<u32>; MAX_PORTS],
    converted: Option<Vec<u32>>,
    conversion_port: Option<usize>,
    state_layout: Option<StateLayout>,
    state: Option<Vec<f32>>,
    scratch: Option<Vec<f32>>,
    needs_scratch: bool,
}

impl Staged {
    fn empty() -> Self {
        Self {
            ports: [PortView {
                value: ValueId(0),
                role: ValueRole::Input,
                dims: Dims::default(),
                source: Source::Scalar,
            }; MAX_PORTS],
            port_count: 0,
            input_count: 0,
            f32_values: core::array::from_fn(|_| None),
            u32_values: core::array::from_fn(|_| None),
            scalars: [None; MAX_PORTS],
            converted: None,
            conversion_port: None,
            state_layout: None,
            state: None,
            scratch: None,
            needs_scratch: false,
        }
    }

    fn resolve(
        ir: &CommandBuffer,
        bindings: &BindingTable,
        buffers: &BufferStore,
        scalars: &ScalarStore,
        command: &Command,
    ) -> Result<Self, ExecutionError> {
        let signature = command.op.code().signature();
        if command.inputs.len() != signature.inputs.len()
            || command.outputs.len() != signature.outputs.len()
        {
            return Err(ExecutionError::InvalidPlan(
                "command arity does not match the operator signature",
            ));
        }

        let input_count = command.inputs.len();
        let port_count = input_count + command.outputs.len();
        if port_count > MAX_PORTS {
            return Err(ExecutionError::InvalidPlan(
                "command has more ports than the reference executor supports",
            ));
        }

        let mut staged = Self::empty();
        staged.input_count = input_count;
        staged.port_count = port_count;

        for (index, &id) in command.inputs.iter().enumerate() {
            staged.add_port(
                ir,
                bindings,
                buffers,
                scalars,
                id,
                signature.inputs[index],
                index,
            )?;
        }
        for (index, &id) in command.outputs.iter().enumerate() {
            staged.add_port(
                ir,
                bindings,
                buffers,
                scalars,
                id,
                signature.outputs[index],
                input_count + index,
            )?;
        }

        let stateful = matches!(command.op, Op::KvAppend { .. } | Op::CausalAttention { .. });
        match command.effect {
            Effect::None => {
                if stateful {
                    return Err(ExecutionError::InvalidPlan(
                        "stateful operator requires a WriteState effect",
                    ));
                }
            }
            Effect::WriteState(id) => {
                if !stateful {
                    return Err(ExecutionError::InvalidPlan(
                        "operator does not write persistent state",
                    ));
                }
                staged.state_layout = Some(resolve_state(ir, bindings, buffers, id)?);
            }
            Effect::ReadState(_) => {
                return Err(ExecutionError::InvalidPlan(
                    "reference executor requires a WriteState effect",
                ));
            }
        }

        if let Some(layout) = staged.state_layout {
            for port in 0..staged.port_count {
                if staged.ports[port].source.buffer_id() == Some(layout.buffer) {
                    return Err(ExecutionError::InvalidPlan(
                        "state buffer aliases a command buffer",
                    ));
                }
            }
        }

        Ok(staged)
    }

    #[allow(clippy::too_many_arguments)]
    fn add_port(
        &mut self,
        ir: &CommandBuffer,
        bindings: &BindingTable,
        buffers: &BufferStore,
        scalars: &ScalarStore,
        id: ValueId,
        spec: PortSpec,
        port: usize,
    ) -> Result<(), ExecutionError> {
        let value = ir
            .values
            .get(id.0 as usize)
            .ok_or(ExecutionError::InvalidPlan(
                "command references an unknown value",
            ))?;
        if !spec.dtypes.contains(value.dtype) {
            return Err(ExecutionError::InvalidPlan(
                "value dtype does not match the operator port",
            ));
        }
        if !spec.roles.contains(value.role) {
            return Err(ExecutionError::InvalidPlan(
                "value role does not match the operator port",
            ));
        }
        let rank = value.shape.rank();
        if rank < spec.min_rank as usize || rank > spec.max_rank as usize {
            return Err(ExecutionError::InvalidPlan(
                "value rank does not match the operator port",
            ));
        }

        let dims = Dims::from_value(value)?;
        let (source, scalar) = match value.role {
            ValueRole::Scalar => {
                let scalar = scalars.get(id).ok_or(ExecutionError::MissingScalar(id))?;
                (Source::Scalar, Some(scalar))
            }
            ValueRole::Temp => {
                if value.dtype != DType::F32 {
                    return Err(ExecutionError::Kernel(KernelError::Unsupported(
                        "scratchpad values must be F32",
                    )));
                }
                let (start, len) = scratch_range(ir, id)?;
                if len != dims.element_count()? {
                    return Err(ExecutionError::InvalidPlan(
                        "slot size does not match the value size",
                    ));
                }
                self.needs_scratch = true;
                (Source::Scratch { start, len }, None)
            }
            _ => {
                let buffer = bindings.get(id).ok_or(ExecutionError::MissingBinding(id))?;
                if buffer == SCRATCHPAD {
                    return Err(ExecutionError::InvalidPlan(
                        "BufferId(0) is reserved for the scratchpad",
                    ));
                }
                let expected = dims.element_count()?;
                match value.dtype {
                    DType::F32 => {
                        let data = buffers
                            .get(buffer)
                            .ok_or(ExecutionError::MissingBuffer(buffer))?;
                        expect_buffer_len(data.len(), expected)?;
                        (Source::ExternalF32(buffer), None)
                    }
                    DType::U32 => {
                        if let Some(data) = buffers.get_u32(buffer) {
                            expect_buffer_len(data.len(), expected)?;
                            (Source::ExternalU32(buffer), None)
                        } else if let Some(data) = buffers.get(buffer) {
                            expect_buffer_len(data.len(), expected)?;
                            if self.conversion_port.is_some() {
                                return Err(ExecutionError::InvalidPlan(
                                    "at most one converted integer tensor is supported per command",
                                ));
                            }
                            self.conversion_port = Some(port);
                            (Source::ExternalU32FromF32(buffer), None)
                        } else {
                            return Err(ExecutionError::MissingBuffer(buffer));
                        }
                    }
                    _ => {
                        return Err(ExecutionError::Kernel(KernelError::Unsupported(
                            "dtype is not supported by the reference executor",
                        )))
                    }
                }
            }
        };

        for previous in 0..port {
            let prior = self.ports[previous];
            if prior.value == id
                && value.role != ValueRole::Scalar
                && prior.role != ValueRole::Scalar
            {
                return Err(ExecutionError::InvalidPlan(
                    "a command aliases one value across multiple ports",
                ));
            }
            if let (Some(left), Some(right)) = (prior.source.buffer_id(), source.buffer_id()) {
                if left == right {
                    return Err(ExecutionError::InvalidPlan(
                        "a command aliases one buffer across multiple ports",
                    ));
                }
            }
        }

        self.ports[port] = PortView {
            value: id,
            role: value.role,
            dims,
            source,
        };
        self.scalars[port] = scalar;
        Ok(())
    }

    /// Takes every external buffer, the state buffer and the scratchpad out of
    /// the store. All of them were validated by [`Self::resolve`].
    fn take(&mut self, buffers: &mut BufferStore) -> Result<(), ExecutionError> {
        for port in 0..self.port_count {
            match self.ports[port].source {
                Source::ExternalF32(id) => {
                    self.f32_values[port] = buffers.take_f32(id);
                    if self.f32_values[port].is_none() {
                        return Err(ExecutionError::MissingBuffer(id));
                    }
                }
                Source::ExternalU32(id) => {
                    self.u32_values[port] = buffers.take_u32(id);
                    if self.u32_values[port].is_none() {
                        return Err(ExecutionError::MissingBuffer(id));
                    }
                }
                Source::ExternalU32FromF32(id) => {
                    self.f32_values[port] = buffers.take_f32(id);
                    if self.f32_values[port].is_none() {
                        return Err(ExecutionError::MissingBuffer(id));
                    }
                }
                Source::Scratch { .. } | Source::Scalar => {}
            }
        }

        if let Some(port) = self.conversion_port {
            let source = self.f32_values[port].take();
            let Some(values) = source.as_deref() else {
                return Err(ExecutionError::InvalidPlan(
                    "converted integer tensor was not staged",
                ));
            };
            if let Err(error) = validate_u32_payload(values) {
                self.f32_values[port] = source;
                return Err(error);
            }
            let mut converted = buffers.take_conversion();
            if converted.capacity() < values.len() {
                buffers.put_conversion(converted);
                self.f32_values[port] = source;
                return Err(ExecutionError::InvalidPlan(
                    "integer tensor conversion exceeds the preallocated buffer; use insert_u32",
                ));
            }
            converted.clear();
            for &value in values {
                converted.push(value as u32);
            }
            self.f32_values[port] = source;
            self.converted = Some(converted);
        }

        if let Some(layout) = self.state_layout {
            self.state = buffers.take_f32(layout.buffer);
            if self.state.is_none() {
                return Err(ExecutionError::MissingBuffer(layout.buffer));
            }
        }

        if self.needs_scratch {
            self.scratch = buffers.take_f32(SCRATCHPAD);
            if self.scratch.is_none() {
                return Err(ExecutionError::MissingBuffer(SCRATCHPAD));
            }
        }

        Ok(())
    }

    /// Puts every taken buffer back into the store.
    fn restore(&mut self, buffers: &mut BufferStore) {
        for port in 0..self.port_count {
            match self.ports[port].source {
                Source::ExternalF32(id) | Source::ExternalU32FromF32(id) => {
                    if let Some(data) = self.f32_values[port].take() {
                        buffers.put_f32(id, data);
                    }
                }
                Source::ExternalU32(id) => {
                    if let Some(data) = self.u32_values[port].take() {
                        buffers.put_u32(id, data);
                    }
                }
                Source::Scratch { .. } | Source::Scalar => {}
            }
        }
        if let Some(data) = self.converted.take() {
            buffers.put_conversion(data);
        }
        if let Some(layout) = self.state_layout {
            if let Some(data) = self.state.take() {
                buffers.put_f32(layout.buffer, data);
            }
        }
        if let Some(data) = self.scratch.take() {
            buffers.put_f32(SCRATCHPAD, data);
        }
    }

    fn input_f32<'a>(
        &'a self,
        temp_parts: &'a [Option<&'a mut [f32]>],
        port: usize,
    ) -> Result<&'a [f32], ExecutionError> {
        if port >= self.port_count || port >= MAX_PORTS {
            return Err(ExecutionError::InvalidPlan("port index is out of range"));
        }
        match self.ports[port].source {
            Source::ExternalF32(_) => self.f32_values[port]
                .as_deref()
                .ok_or(ExecutionError::InvalidPlan("input buffer was not staged")),
            Source::Scratch { .. } => {
                temp_parts[port]
                    .as_deref()
                    .ok_or(ExecutionError::InvalidPlan(
                        "scratchpad slot was not staged",
                    ))
            }
            _ => Err(ExecutionError::InvalidPlan("port is not an F32 input")),
        }
    }

    fn input_u32(&self, port: usize) -> Result<&[u32], ExecutionError> {
        if port >= self.port_count || port >= MAX_PORTS {
            return Err(ExecutionError::InvalidPlan("port index is out of range"));
        }
        match self.ports[port].source {
            Source::ExternalU32(_) => {
                self.u32_values[port]
                    .as_deref()
                    .ok_or(ExecutionError::InvalidPlan(
                        "integer input buffer was not staged",
                    ))
            }
            Source::ExternalU32FromF32(_) => self.converted.as_deref().ok_or(
                ExecutionError::InvalidPlan("converted integer tensor was not staged"),
            ),
            _ => Err(ExecutionError::InvalidPlan("port is not a U32 input")),
        }
    }

    fn scalar(&self, port: usize) -> Result<u32, ExecutionError> {
        if port >= self.port_count || port >= MAX_PORTS {
            return Err(ExecutionError::InvalidPlan("port index is out of range"));
        }
        self.scalars[port].ok_or(ExecutionError::InvalidPlan("scalar port was not staged"))
    }

    fn input_dims(&self, port: usize) -> Result<Dims, ExecutionError> {
        if port >= self.port_count || port >= MAX_PORTS {
            return Err(ExecutionError::InvalidPlan("port index is out of range"));
        }
        Ok(self.ports[port].dims)
    }

    fn output_dims(&self) -> Result<Dims, ExecutionError> {
        if self.input_count >= self.port_count {
            return Err(invalid_shape("operator requires an output"));
        }
        self.input_dims(self.input_count)
    }
}

/// Destination of one command's single output or the absence of one.
enum OutputStorage<'a> {
    External(Option<Vec<f32>>),
    Scratch(Option<&'a mut [f32]>),
}

impl OutputStorage<'_> {
    fn as_mut_slice(&mut self) -> Result<&mut [f32], ExecutionError> {
        match self {
            Self::External(data) => data
                .as_deref_mut()
                .ok_or(ExecutionError::InvalidPlan("output buffer was not staged")),
            Self::Scratch(piece) => piece.as_deref_mut().ok_or(ExecutionError::InvalidPlan(
                "scratchpad output was not staged",
            )),
        }
    }
}

fn take_output<'a>(
    staged: &mut Staged,
    temp_parts: &mut [Option<&'a mut [f32]>; MAX_PORTS],
    port: usize,
) -> Result<OutputStorage<'a>, ExecutionError> {
    if port >= staged.port_count || port >= MAX_PORTS {
        return Err(ExecutionError::InvalidPlan(
            "output port index is out of range",
        ));
    }
    match staged.ports[port].source {
        Source::ExternalF32(_) => Ok(OutputStorage::External(staged.f32_values[port].take())),
        Source::Scratch { .. } => Ok(OutputStorage::Scratch(temp_parts[port].take())),
        _ => Err(ExecutionError::InvalidPlan(
            "operator output must be an F32 or scratchpad value",
        )),
    }
}

fn restore_output<'a>(
    staged: &mut Staged,
    temp_parts: &mut [Option<&'a mut [f32]>; MAX_PORTS],
    port: usize,
    output: OutputStorage<'a>,
) {
    match output {
        OutputStorage::External(data) => staged.f32_values[port] = data,
        OutputStorage::Scratch(piece) => temp_parts[port] = piece,
    }
}

/// Runs one prepared command, restoring its output slot on every path.
macro_rules! with_output {
    ($staged:expr, $temp:expr, $port:expr, |$output:ident| $body:block) => {{
        let mut output_storage = take_output($staged, $temp, $port)?;
        let result = (|| -> Result<(), ExecutionError> {
            let $output = output_storage.as_mut_slice()?;
            $body
        })();
        restore_output($staged, $temp, $port, output_storage);
        result
    }};
}

fn execute_command(
    ir: &CommandBuffer,
    bindings: &BindingTable,
    buffers: &mut BufferStore,
    scalars: &ScalarStore,
    command: &Command,
) -> Result<(), ExecutionError> {
    let mut staged = Staged::resolve(ir, bindings, buffers, scalars, command)?;
    if let Err(error) = staged.take(buffers) {
        staged.restore(buffers);
        return Err(error);
    }
    let result = run_command(command, &mut staged);
    staged.restore(buffers);
    result
}

fn run_command(command: &Command, staged: &mut Staged) -> Result<(), ExecutionError> {
    let mut scratch = staged.scratch.take();
    let result = match scratch.as_mut() {
        Some(data) => match split_scratch(data, &staged.ports, staged.port_count) {
            Ok(mut temp_parts) => dispatch(command, staged, &mut temp_parts),
            Err(error) => Err(error),
        },
        None => {
            let mut temp_parts: [Option<&mut [f32]>; MAX_PORTS] = core::array::from_fn(|_| None);
            dispatch(command, staged, &mut temp_parts)
        }
    };
    staged.scratch = scratch;
    result
}

fn dispatch(
    command: &Command,
    staged: &mut Staged,
    temp_parts: &mut [Option<&mut [f32]>; MAX_PORTS],
) -> Result<(), ExecutionError> {
    let output_port = staged.input_count;
    match command.op {
        Op::MatMulNT => {
            let a = staged.input_dims(0)?;
            let b = staged.input_dims(1)?;
            let out = staged.output_dims()?;
            let (batch_a, batch_b, m, k, n) = matmul_geometry(&a, &b, &out, true)?;
            with_output!(staged, temp_parts, output_port, |output| {
                let left = staged.input_f32(temp_parts, 0)?;
                let right = staged.input_f32(temp_parts, 1)?;
                matmul_nt_broadcast(left, right, output, batch_a, batch_b, m, k, n)
            })
        }
        Op::MatMulNN => {
            let a = staged.input_dims(0)?;
            let b = staged.input_dims(1)?;
            let out = staged.output_dims()?;
            let (batch_a, batch_b, m, k, n) = matmul_geometry(&a, &b, &out, false)?;
            with_output!(staged, temp_parts, output_port, |output| {
                let left = staged.input_f32(temp_parts, 0)?;
                let right = staged.input_f32(temp_parts, 1)?;
                matmul_nn_broadcast(left, right, output, batch_a, batch_b, m, k, n)
            })
        }
        Op::Softmax => {
            let input = staged.input_dims(0)?;
            let out = staged.output_dims()?;
            let (rows, cols) = softmax_geometry(&input, &out)?;
            if rows == 0 || cols == 0 {
                return Err(invalid_shape("softmax requires nonzero rows and cols"));
            }
            with_output!(staged, temp_parts, output_port, |output| {
                let values = staged.input_f32(temp_parts, 0)?;
                copy_into(values, output)?;
                reference::softmax(output, rows, cols).map_err(ExecutionError::Kernel)
            })
        }
        Op::SoftmaxCausal => {
            let input = staged.input_dims(0)?;
            let out = staged.output_dims()?;
            let (rows, cols) = softmax_geometry(&input, &out)?;
            let seq = scalar_usize(staged, 1)?;
            let past_len = scalar_usize(staged, 2)?;
            if rows == 0 || cols == 0 {
                return Err(invalid_shape(
                    "softmax_causal requires nonzero rows and cols",
                ));
            }
            if seq == 0 {
                return Err(invalid_shape("softmax_causal requires a nonzero seq"));
            }
            let expected_cols = past_len
                .checked_add(seq)
                .ok_or(ExecutionError::Kernel(KernelError::ShapeOverflow))?;
            if cols != expected_cols {
                return Err(invalid_shape(
                    "softmax_causal requires cols == past_len + seq",
                ));
            }
            if rows % seq != 0 {
                return Err(invalid_shape(
                    "softmax_causal requires rows to be a multiple of seq",
                ));
            }
            with_output!(staged, temp_parts, output_port, |output| {
                let values = staged.input_f32(temp_parts, 0)?;
                copy_into(values, output)?;
                reference::softmax_causal(output, rows, cols, seq, past_len)
                    .map_err(ExecutionError::Kernel)
            })
        }
        Op::RmsNorm { eps } => {
            let input = staged.input_dims(0)?;
            let weight = staged.input_dims(1)?;
            let out = staged.output_dims()?;
            let (rows, hidden) = row_geometry(&input)?;
            if weight.rank != 1 || weight.dim(0)? != hidden {
                return Err(invalid_shape("rms_norm weight shape must be [hidden]"));
            }
            if !input.same_as(&out) {
                return Err(invalid_shape("rms_norm output shape must match its input"));
            }
            with_output!(staged, temp_parts, output_port, |output| {
                let values = staged.input_f32(temp_parts, 0)?;
                let scales = staged.input_f32(temp_parts, 1)?;
                reference::rms_norm(values, scales, output, rows, hidden, eps)
                    .map_err(ExecutionError::Kernel)
            })
        }
        Op::Rope { variant, theta } => {
            let input = staged.input_dims(0)?;
            let positions = staged.input_dims(1)?;
            let out = staged.output_dims()?;
            let (batch, heads, seq, head_dim) = rope_geometry(&input, &positions)?;
            if !input.same_as(&out) {
                return Err(invalid_shape("rope output shape must match its input"));
            }
            with_output!(staged, temp_parts, output_port, |output| {
                let values = staged.input_f32(temp_parts, 0)?;
                let positions = staged.input_u32(1)?;
                copy_into(values, output)?;
                match variant {
                    sl_ir::RopeVariant::Neox => {
                        reference::rope_neox(output, positions, batch, heads, seq, head_dim, theta)
                    }
                    sl_ir::RopeVariant::GptJ => {
                        reference::rope_gptj(output, positions, batch, heads, seq, head_dim, theta)
                    }
                }
                .map_err(ExecutionError::Kernel)
            })
        }
        Op::Silu => {
            let input = staged.input_dims(0)?;
            let out = staged.output_dims()?;
            if !input.same_as(&out) {
                return Err(invalid_shape("silu output shape must match its input"));
            }
            with_output!(staged, temp_parts, output_port, |output| {
                let values = staged.input_f32(temp_parts, 0)?;
                copy_into(values, output)?;
                reference::silu(output).map_err(ExecutionError::Kernel)
            })
        }
        Op::Mul => {
            let a = staged.input_dims(0)?;
            let b = staged.input_dims(1)?;
            let out = staged.output_dims()?;
            if !a.same_as(&b) || !a.same_as(&out) {
                return Err(invalid_shape(
                    "mul operands and output must have identical shapes",
                ));
            }
            let len = a.element_count()?;
            with_output!(staged, temp_parts, output_port, |output| {
                let left = staged.input_f32(temp_parts, 0)?;
                let right = staged.input_f32(temp_parts, 1)?;
                reference::mul(left, right, output, len).map_err(ExecutionError::Kernel)
            })
        }
        Op::Add => {
            let a = staged.input_dims(0)?;
            let b = staged.input_dims(1)?;
            let out = staged.output_dims()?;
            if !a.same_as(&b) || !a.same_as(&out) {
                return Err(invalid_shape(
                    "add operands and output must have identical shapes",
                ));
            }
            let len = a.element_count()?;
            with_output!(staged, temp_parts, output_port, |output| {
                let left = staged.input_f32(temp_parts, 0)?;
                let right = staged.input_f32(temp_parts, 1)?;
                reference::add(left, right, output, len).map_err(ExecutionError::Kernel)
            })
        }
        Op::Scale { factor } => {
            let input = staged.input_dims(0)?;
            let out = staged.output_dims()?;
            if !input.same_as(&out) {
                return Err(invalid_shape("scale output shape must match its input"));
            }
            with_output!(staged, temp_parts, output_port, |output| {
                let values = staged.input_f32(temp_parts, 0)?;
                copy_into(values, output)?;
                reference::scale(output, factor).map_err(ExecutionError::Kernel)
            })
        }
        Op::Gather => {
            let table = staged.input_dims(0)?;
            let ids = staged.input_dims(1)?;
            let out = staged.output_dims()?;
            if table.rank != 2 {
                return Err(invalid_shape(
                    "gather table must have shape [vocab, hidden]",
                ));
            }
            let vocab = table.dim(0)?;
            let hidden = table.dim(1)?;
            let rows = ids.element_count()?;
            if out.rank != ids.rank + 1 {
                return Err(invalid_shape(
                    "gather output rank must be the id rank plus one",
                ));
            }
            for index in 0..ids.rank {
                if out.dim(index)? != ids.dim(index)? {
                    return Err(invalid_shape(
                        "gather output prefix must match the id shape",
                    ));
                }
            }
            if out.last()? != hidden {
                return Err(invalid_shape("gather output width must equal hidden"));
            }
            with_output!(staged, temp_parts, output_port, |output| {
                let table = staged.input_f32(temp_parts, 0)?;
                let ids = staged.input_u32(1)?;
                reference::gather(table, ids, output, vocab, hidden, rows)
                    .map_err(ExecutionError::Kernel)
            })
        }
        Op::LastToken => {
            let input = staged.input_dims(0)?;
            let out = staged.output_dims()?;
            let seq_len = scalar_usize(staged, 1)?;
            if input.rank != 3 || out.rank != 2 {
                return Err(invalid_shape(
                    "last_token shapes must be [B, M, H] -> [B, H]",
                ));
            }
            let batch = input.dim(0)?;
            let seq = input.dim(1)?;
            let hidden = input.dim(2)?;
            if out.dim(0)? != batch || out.dim(1)? != hidden {
                return Err(invalid_shape(
                    "last_token output shape must be [batch, hidden]",
                ));
            }
            with_output!(staged, temp_parts, output_port, |output| {
                let values = staged.input_f32(temp_parts, 0)?;
                reference::last_token(values, output, batch, seq, hidden, seq_len)
                    .map_err(ExecutionError::Kernel)
            })
        }
        Op::KvAppend { .. } => {
            let k = staged.input_dims(0)?;
            let v = staged.input_dims(1)?;
            let past_len = scalar_usize(staged, 2)?;
            let layer = op_layer(command)?;
            let layout = staged.state_layout.ok_or(ExecutionError::InvalidPlan(
                "kv_append requires a WriteState effect",
            ))?;
            let (batch, kv_heads, seq, head_dim) = attention_kv_geometry(&k, &v)?;
            check_state_geometry(&layout, batch, kv_heads, head_dim)?;
            if layer >= layout.layers {
                return Err(invalid_shape("kv_append layer is out of range"));
            }
            let mut state = staged.state.take();
            let result = (|| -> Result<(), ExecutionError> {
                let cache = state.as_deref_mut().ok_or(ExecutionError::InvalidPlan(
                    "kv_append state buffer was not staged",
                ))?;
                let keys = staged.input_f32(temp_parts, 0)?;
                let values = staged.input_f32(temp_parts, 1)?;
                reference::kv_append(
                    keys,
                    values,
                    cache,
                    batch,
                    layout.layers,
                    layer,
                    kv_heads,
                    head_dim,
                    layout.max_seq,
                    past_len,
                    seq,
                )
                .map_err(ExecutionError::Kernel)
            })();
            staged.state = state;
            result
        }
        Op::SplitHeads { heads, head_dim } => {
            let input = staged.input_dims(0)?;
            let out = staged.output_dims()?;
            let heads = usize::try_from(heads)
                .map_err(|_| ExecutionError::InvalidPlan("heads does not fit in usize"))?;
            let head_dim = usize::try_from(head_dim)
                .map_err(|_| ExecutionError::InvalidPlan("head_dim does not fit in usize"))?;
            let (batch, seq) = split_geometry(&input, heads, head_dim)?;
            if out.rank != 4
                || out.dim(0)? != batch
                || out.dim(1)? != heads
                || out.dim(2)? != seq
                || out.dim(3)? != head_dim
            {
                return Err(invalid_shape(
                    "split_heads output shape must be [batch, heads, seq, head_dim]",
                ));
            }
            with_output!(staged, temp_parts, output_port, |output| {
                let values = staged.input_f32(temp_parts, 0)?;
                reference::split_heads(values, output, batch, seq, heads, head_dim)
                    .map_err(ExecutionError::Kernel)
            })
        }
        Op::MergeHeads { heads, head_dim } => {
            let input = staged.input_dims(0)?;
            let out = staged.output_dims()?;
            let heads = usize::try_from(heads)
                .map_err(|_| ExecutionError::InvalidPlan("heads does not fit in usize"))?;
            let head_dim = usize::try_from(head_dim)
                .map_err(|_| ExecutionError::InvalidPlan("head_dim does not fit in usize"))?;
            if input.rank != 4 {
                return Err(invalid_shape(
                    "merge_heads input shape must be [batch, heads, seq, head_dim]",
                ));
            }
            let batch = input.dim(0)?;
            let seq = input.dim(2)?;
            if input.dim(1)? != heads || input.dim(3)? != head_dim {
                return Err(invalid_shape(
                    "merge_heads input does not match its heads/head_dim attributes",
                ));
            }
            let width = heads
                .checked_mul(head_dim)
                .ok_or(ExecutionError::Kernel(KernelError::ShapeOverflow))?;
            if out.rank != 3 || out.dim(0)? != batch || out.dim(1)? != seq || out.dim(2)? != width {
                return Err(invalid_shape(
                    "merge_heads output shape must be [batch, seq, heads * head_dim]",
                ));
            }
            with_output!(staged, temp_parts, output_port, |output| {
                let values = staged.input_f32(temp_parts, 0)?;
                reference::merge_heads(values, output, batch, seq, heads, head_dim)
                    .map_err(ExecutionError::Kernel)
            })
        }
        Op::CausalAttention {
            heads,
            kv_heads,
            head_dim,
            scale,
            ..
        } => {
            let q = staged.input_dims(0)?;
            let k = staged.input_dims(1)?;
            let v = staged.input_dims(2)?;
            let out = staged.output_dims()?;
            let past_len = scalar_usize(staged, 3)?;
            let layer = op_layer(command)?;
            let layout = staged.state_layout.ok_or(ExecutionError::InvalidPlan(
                "causal_attention requires a WriteState effect",
            ))?;
            let heads = usize::try_from(heads)
                .map_err(|_| ExecutionError::InvalidPlan("heads does not fit in usize"))?;
            let kv_heads = usize::try_from(kv_heads)
                .map_err(|_| ExecutionError::InvalidPlan("kv_heads does not fit in usize"))?;
            let head_dim = usize::try_from(head_dim)
                .map_err(|_| ExecutionError::InvalidPlan("head_dim does not fit in usize"))?;
            if q.rank != 4 || k.rank != 4 || v.rank != 4 || out.rank != 4 {
                return Err(invalid_shape("causal_attention operands must have rank 4"));
            }
            let batch = q.dim(0)?;
            let query_heads = q.dim(1)?;
            let seq = q.dim(2)?;
            let width = q.dim(3)?;
            if query_heads != heads || width != head_dim {
                return Err(invalid_shape(
                    "causal_attention query shape does not match its attributes",
                ));
            }
            if k.dim(0)? != batch
                || v.dim(0)? != batch
                || k.dim(1)? != kv_heads
                || v.dim(1)? != kv_heads
                || k.dim(2)? != seq
                || v.dim(2)? != seq
                || k.dim(3)? != head_dim
                || v.dim(3)? != head_dim
            {
                return Err(invalid_shape(
                    "causal_attention key/value shapes must be [batch, kv_heads, seq, head_dim]",
                ));
            }
            if !out.same_as(&q) {
                return Err(invalid_shape(
                    "causal_attention output shape must match the query shape",
                ));
            }
            check_state_geometry(&layout, batch, kv_heads, head_dim)?;
            if layer >= layout.layers {
                return Err(invalid_shape("causal_attention layer is out of range"));
            }
            let mut state = staged.state.take();
            let mut output = take_output(staged, temp_parts, output_port)?;
            let result = (|| -> Result<(), ExecutionError> {
                let cache = state.as_deref_mut().ok_or(ExecutionError::InvalidPlan(
                    "causal_attention state buffer was not staged",
                ))?;
                let queries = staged.input_f32(temp_parts, 0)?;
                let keys = staged.input_f32(temp_parts, 1)?;
                let values = staged.input_f32(temp_parts, 2)?;
                let output = output.as_mut_slice()?;
                reference::causal_attention(
                    queries,
                    keys,
                    values,
                    cache,
                    output,
                    batch,
                    layout.layers,
                    layer,
                    heads,
                    kv_heads,
                    seq,
                    past_len,
                    head_dim,
                    layout.max_seq,
                    scale,
                )
                .map_err(ExecutionError::Kernel)
            })();
            restore_output(staged, temp_parts, output_port, output);
            staged.state = state;
            result
        }
        Op::SwiGlu => {
            let input = staged.input_dims(0)?;
            let gate = staged.input_dims(1)?;
            let up = staged.input_dims(2)?;
            let down = staged.input_dims(3)?;
            let out = staged.output_dims()?;
            let (rows, hidden) = row_geometry(&input)?;
            if gate.rank != 2 || up.rank != 2 || down.rank != 2 {
                return Err(invalid_shape("swiglu weights must have rank 2"));
            }
            let inter = gate.dim(0)?;
            if gate.dim(1)? != hidden
                || up.dim(0)? != inter
                || up.dim(1)? != hidden
                || down.dim(0)? != hidden
                || down.dim(1)? != inter
            {
                return Err(invalid_shape(
                    "swiglu weight shapes must be [inter, hidden], [inter, hidden], [hidden, inter]",
                ));
            }
            if !input.same_as(&out) {
                return Err(invalid_shape("swiglu output shape must match its input"));
            }
            with_output!(staged, temp_parts, output_port, |output| {
                let values = staged.input_f32(temp_parts, 0)?;
                let gate = staged.input_f32(temp_parts, 1)?;
                let up = staged.input_f32(temp_parts, 2)?;
                let down = staged.input_f32(temp_parts, 3)?;
                reference::swiglu(values, gate, up, down, output, rows, hidden, inter)
                    .map_err(ExecutionError::Kernel)
            })
        }
        Op::RmsNormMatMul { eps } => {
            let input = staged.input_dims(0)?;
            let weight = staged.input_dims(1)?;
            let matrix = staged.input_dims(2)?;
            let out = staged.output_dims()?;
            let (rows, hidden) = row_geometry(&input)?;
            if weight.rank != 1 || weight.dim(0)? != hidden {
                return Err(invalid_shape(
                    "rms_norm_matmul weight shape must be [hidden]",
                ));
            }
            if matrix.rank != 2 || matrix.dim(1)? != hidden {
                return Err(invalid_shape(
                    "rms_norm_matmul matrix shape must be [out_features, hidden]",
                ));
            }
            let out_features = matrix.dim(0)?;
            if out.rank != input.rank {
                return Err(invalid_shape(
                    "rms_norm_matmul output rank must match its input rank",
                ));
            }
            for index in 0..input.rank - 1 {
                if out.dim(index)? != input.dim(index)? {
                    return Err(invalid_shape(
                        "rms_norm_matmul output prefix must match its input",
                    ));
                }
            }
            if out.last()? != out_features {
                return Err(invalid_shape(
                    "rms_norm_matmul output width must equal out_features",
                ));
            }
            with_output!(staged, temp_parts, output_port, |output| {
                let values = staged.input_f32(temp_parts, 0)?;
                let weight = staged.input_f32(temp_parts, 1)?;
                let matrix = staged.input_f32(temp_parts, 2)?;
                reference::rms_norm_matmul(
                    values,
                    weight,
                    matrix,
                    output,
                    rows,
                    hidden,
                    out_features,
                    eps,
                )
                .map_err(ExecutionError::Kernel)
            })
        }
    }
}

fn matmul_geometry(
    a: &Dims,
    b: &Dims,
    out: &Dims,
    transposed: bool,
) -> Result<(usize, usize, usize, usize, usize), ExecutionError> {
    if a.rank < 2 || b.rank < 2 || out.rank < 2 {
        return Err(invalid_shape(
            "matmul operands and output must have rank at least 2",
        ));
    }
    let m = a.dim(a.rank - 2)?;
    let k = a.dim(a.rank - 1)?;
    let (b_k, n) = if transposed {
        (b.dim(b.rank - 1)?, b.dim(b.rank - 2)?)
    } else {
        (b.dim(b.rank - 2)?, b.dim(b.rank - 1)?)
    };
    if k != b_k {
        return Err(invalid_shape("matmul reduction dimensions must match"));
    }
    let batch_a = a.product(a.rank - 2)?;
    let batch_b = b.product(b.rank - 2)?;
    let batch = broadcast_batch(batch_a, batch_b)?;
    if out.dim(out.rank - 2)? != m || out.dim(out.rank - 1)? != n {
        return Err(invalid_shape("matmul output shape must be m x n"));
    }
    if out.product(out.rank - 2)? != batch {
        return Err(invalid_shape("matmul output batch must match the operands"));
    }
    Ok((batch_a, batch_b, m, k, n))
}

/// Broadcasts two matmul prefix products under `equal-or-one` rules.
///
/// A zero-length batch only broadcasts with `0` or `1`, mirroring NumPy: shape
/// `[0, M, K]` times `[N, K]` yields `[0, N]`, while `[0, M, K]` times
/// `[5, N, K]` is rejected.
fn broadcast_batch(left: usize, right: usize) -> Result<usize, ExecutionError> {
    if left == right {
        return Ok(left);
    }
    if left == 0 || right == 0 {
        if left.max(right) <= 1 {
            return Ok(0);
        }
        return Err(invalid_shape(
            "matmul batch dimensions are not broadcastable",
        ));
    }
    if left == 1 {
        return Ok(right);
    }
    if right == 1 {
        return Ok(left);
    }
    Err(invalid_shape(
        "matmul batch dimensions are not broadcastable",
    ))
}

#[allow(clippy::too_many_arguments)]
fn matmul_nt_broadcast(
    left: &[f32],
    right: &[f32],
    output: &mut [f32],
    batch_a: usize,
    batch_b: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), ExecutionError> {
    let batch = batch_a.max(batch_b);
    if batch == 0 {
        return Ok(());
    }
    if batch_a == batch_b {
        return reference::matmul_nt(left, right, output, batch, m, k, n)
            .map_err(ExecutionError::Kernel);
    }
    let a_stride = m
        .checked_mul(k)
        .ok_or(ExecutionError::Kernel(KernelError::ShapeOverflow))?;
    let b_stride = n
        .checked_mul(k)
        .ok_or(ExecutionError::Kernel(KernelError::ShapeOverflow))?;
    let out_stride = m
        .checked_mul(n)
        .ok_or(ExecutionError::Kernel(KernelError::ShapeOverflow))?;
    for index in 0..batch {
        let a_start = if batch_a == 1 { 0 } else { index * a_stride };
        let b_start = if batch_b == 1 { 0 } else { index * b_stride };
        let out_start = index * out_stride;
        let a_slice = left
            .get(a_start..a_start + a_stride)
            .ok_or_else(|| invalid_shape("matmul left operand is too short"))?;
        let b_slice = right
            .get(b_start..b_start + b_stride)
            .ok_or_else(|| invalid_shape("matmul right operand is too short"))?;
        let out_slice = output
            .get_mut(out_start..out_start + out_stride)
            .ok_or_else(|| invalid_shape("matmul output is too short"))?;
        reference::matmul_nt(a_slice, b_slice, out_slice, 1, m, k, n)
            .map_err(ExecutionError::Kernel)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn matmul_nn_broadcast(
    left: &[f32],
    right: &[f32],
    output: &mut [f32],
    batch_a: usize,
    batch_b: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), ExecutionError> {
    let batch = batch_a.max(batch_b);
    if batch == 0 {
        return Ok(());
    }
    if batch_a == batch_b {
        return reference::matmul_nn(left, right, output, batch, m, k, n)
            .map_err(ExecutionError::Kernel);
    }
    let a_stride = m
        .checked_mul(k)
        .ok_or(ExecutionError::Kernel(KernelError::ShapeOverflow))?;
    let b_stride = k
        .checked_mul(n)
        .ok_or(ExecutionError::Kernel(KernelError::ShapeOverflow))?;
    let out_stride = m
        .checked_mul(n)
        .ok_or(ExecutionError::Kernel(KernelError::ShapeOverflow))?;
    for index in 0..batch {
        let a_start = if batch_a == 1 { 0 } else { index * a_stride };
        let b_start = if batch_b == 1 { 0 } else { index * b_stride };
        let out_start = index * out_stride;
        let a_slice = left
            .get(a_start..a_start + a_stride)
            .ok_or_else(|| invalid_shape("matmul left operand is too short"))?;
        let b_slice = right
            .get(b_start..b_start + b_stride)
            .ok_or_else(|| invalid_shape("matmul right operand is too short"))?;
        let out_slice = output
            .get_mut(out_start..out_start + out_stride)
            .ok_or_else(|| invalid_shape("matmul output is too short"))?;
        reference::matmul_nn(a_slice, b_slice, out_slice, 1, m, k, n)
            .map_err(ExecutionError::Kernel)?;
    }
    Ok(())
}

fn softmax_geometry(input: &Dims, out: &Dims) -> Result<(usize, usize), ExecutionError> {
    if !input.same_as(out) {
        return Err(invalid_shape("softmax output shape must match its input"));
    }
    let cols = input.last()?;
    let rows = input.product(input.rank - 1)?;
    Ok((rows, cols))
}

fn row_geometry(input: &Dims) -> Result<(usize, usize), ExecutionError> {
    // Rank-1 `[H]` is a single row; it is valid for `RmsNorm`. The other
    // callers have a signature-level minimum rank of two.
    let hidden = input.last()?;
    let rows = input.product(input.rank - 1)?;
    Ok((rows, hidden))
}

fn rope_geometry(
    input: &Dims,
    positions: &Dims,
) -> Result<(usize, usize, usize, usize), ExecutionError> {
    let (batch, heads, seq, head_dim) = match input.rank {
        4 => (input.dim(0)?, input.dim(1)?, input.dim(2)?, input.dim(3)?),
        3 => (input.dim(0)?, 1, input.dim(1)?, input.dim(2)?),
        _ => return Err(invalid_shape("rope input must have rank 3 or 4")),
    };
    if positions.rank != 2 || positions.dim(0)? != batch || positions.dim(1)? != seq {
        return Err(invalid_shape("rope positions shape must be [batch, seq]"));
    }
    Ok((batch, heads, seq, head_dim))
}

fn split_geometry(
    input: &Dims,
    heads: usize,
    head_dim: usize,
) -> Result<(usize, usize), ExecutionError> {
    if input.rank != 3 {
        return Err(invalid_shape(
            "split_heads input shape must be [batch, seq, heads * head_dim]",
        ));
    }
    let batch = input.dim(0)?;
    let seq = input.dim(1)?;
    let width = heads
        .checked_mul(head_dim)
        .ok_or(ExecutionError::Kernel(KernelError::ShapeOverflow))?;
    if input.dim(2)? != width {
        return Err(invalid_shape(
            "split_heads input width must equal heads * head_dim",
        ));
    }
    Ok((batch, seq))
}

fn attention_kv_geometry(
    k: &Dims,
    v: &Dims,
) -> Result<(usize, usize, usize, usize), ExecutionError> {
    if k.rank != 4 || v.rank != 4 {
        return Err(invalid_shape("kv_append key/value must have rank 4"));
    }
    if !k.same_as(v) {
        return Err(invalid_shape("kv_append key and value shapes must match"));
    }
    Ok((k.dim(0)?, k.dim(1)?, k.dim(2)?, k.dim(3)?))
}

fn check_state_geometry(
    layout: &StateLayout,
    batch: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<(), ExecutionError> {
    if layout.batch != batch || layout.kv_heads != kv_heads || layout.head_dim != head_dim {
        return Err(invalid_shape(
            "KV cache geometry does not match the operator input",
        ));
    }
    Ok(())
}

fn op_layer(command: &Command) -> Result<usize, ExecutionError> {
    let layer = command
        .op
        .layer()
        .ok_or(ExecutionError::InvalidPlan("operator requires a layer"))?;
    usize::try_from(layer).map_err(|_| ExecutionError::InvalidPlan("layer does not fit in usize"))
}

fn scalar_usize(staged: &Staged, port: usize) -> Result<usize, ExecutionError> {
    let scalar = staged.scalar(port)?;
    usize::try_from(scalar).map_err(|_| ExecutionError::InvalidPlan("scalar does not fit in usize"))
}

fn copy_into(input: &[f32], output: &mut [f32]) -> Result<(), ExecutionError> {
    if input.len() != output.len() {
        return Err(ExecutionError::Kernel(KernelError::LengthMismatch {
            expected: input.len(),
            actual: output.len(),
        }));
    }
    output.copy_from_slice(input);
    Ok(())
}

fn expect_buffer_len(actual: usize, expected: usize) -> Result<(), ExecutionError> {
    if actual == expected {
        Ok(())
    } else {
        Err(ExecutionError::Kernel(KernelError::LengthMismatch {
            expected,
            actual,
        }))
    }
}

fn invalid_shape(reason: &'static str) -> ExecutionError {
    ExecutionError::Kernel(KernelError::InvalidShape(reason))
}

fn scratch_range(ir: &CommandBuffer, id: ValueId) -> Result<(usize, usize), ExecutionError> {
    let slot =
        ir.plan
            .slots
            .iter()
            .find(|slot| slot.value == id)
            .ok_or(ExecutionError::InvalidPlan(
                "Temp value is missing from the memory plan",
            ))?;
    if slot.offset % 4 != 0 {
        return Err(ExecutionError::InvalidPlan(
            "slot offset is not a multiple of 4",
        ));
    }
    if slot.bytes % 4 != 0 {
        return Err(ExecutionError::InvalidPlan(
            "slot size is not a multiple of 4",
        ));
    }
    let start = usize::try_from(slot.offset / 4)
        .map_err(|_| ExecutionError::InvalidPlan("slot offset does not fit in usize"))?;
    let len = usize::try_from(slot.bytes / 4)
        .map_err(|_| ExecutionError::InvalidPlan("slot size does not fit in usize"))?;
    Ok((start, len))
}

fn validate_plan(ir: &CommandBuffer, buffers: &BufferStore) -> Result<(), ExecutionError> {
    let scratch_len = buffers.get(SCRATCHPAD).map(<[f32]>::len);
    for (index, slot) in ir.plan.slots.iter().enumerate() {
        if ir.plan.slots[..index]
            .iter()
            .any(|other| other.value == slot.value)
        {
            return Err(ExecutionError::InvalidPlan(
                "a value has more than one scratchpad slot",
            ));
        }
        let value = ir
            .values
            .get(slot.value.0 as usize)
            .ok_or(ExecutionError::InvalidPlan(
                "scratchpad slot references an unknown value",
            ))?;
        if value.role != ValueRole::Temp {
            return Err(ExecutionError::InvalidPlan(
                "only Temp values may occupy scratchpad slots",
            ));
        }
        if value.dtype != DType::F32 {
            return Err(ExecutionError::Kernel(KernelError::Unsupported(
                "scratchpad values must be F32",
            )));
        }
        if slot.offset % 4 != 0 {
            return Err(ExecutionError::InvalidPlan(
                "slot offset is not a multiple of 4",
            ));
        }
        if slot.bytes % 4 != 0 {
            return Err(ExecutionError::InvalidPlan(
                "slot size is not a multiple of 4",
            ));
        }
        let end = slot
            .offset
            .checked_add(slot.bytes)
            .ok_or(ExecutionError::InvalidPlan("slot range overflows"))?;
        if end > ir.plan.scratchpad_bytes {
            return Err(ExecutionError::InvalidPlan("slot exceeds scratchpad"));
        }
        let expected = value
            .byte_size()
            .map_err(|_| ExecutionError::InvalidPlan("value byte size overflows"))?;
        if slot.bytes != expected {
            return Err(ExecutionError::InvalidPlan(
                "slot size does not match the value size",
            ));
        }
        let Some(scratch_len) = scratch_len else {
            return Err(ExecutionError::MissingBuffer(SCRATCHPAD));
        };
        let scratch_bytes = u64::try_from(scratch_len)
            .map_err(|_| ExecutionError::InvalidPlan("scratchpad size does not fit in u64"))?
            .checked_mul(4)
            .ok_or(ExecutionError::InvalidPlan("scratchpad size overflows"))?;
        if end > scratch_bytes {
            return Err(ExecutionError::InvalidPlan(
                "slot exceeds scratchpad buffer",
            ));
        }
    }
    Ok(())
}

fn resolve_state(
    ir: &CommandBuffer,
    bindings: &BindingTable,
    buffers: &BufferStore,
    value: ValueId,
) -> Result<StateLayout, ExecutionError> {
    let state_value = ir
        .values
        .get(value.0 as usize)
        .ok_or(ExecutionError::InvalidPlan(
            "state effect references an unknown value",
        ))?;
    if state_value.role != ValueRole::State {
        return Err(ExecutionError::InvalidPlan(
            "state effect target is not a State value",
        ));
    }
    let declaration = ir
        .states
        .iter()
        .find(|declaration| declaration.value == value)
        .ok_or(ExecutionError::InvalidPlan("state value has no StateDecl"))?;
    let StateKind::KvCache {
        batch,
        layers,
        kv_heads,
        head_dim,
        max_seq,
        dtype,
    } = declaration.kind;
    if dtype != DType::F32 {
        return Err(ExecutionError::Kernel(KernelError::Unsupported(
            "reference executor only supports F32 KV caches",
        )));
    }
    let max_seq = max_seq.as_const().ok_or(ExecutionError::InvalidPlan(
        "KV cache max_seq is still symbolic",
    ))?;
    let batch = usize::try_from(batch)
        .map_err(|_| ExecutionError::InvalidPlan("batch does not fit in usize"))?;
    let layers = usize::try_from(layers)
        .map_err(|_| ExecutionError::InvalidPlan("layers do not fit in usize"))?;
    let kv_heads = usize::try_from(kv_heads)
        .map_err(|_| ExecutionError::InvalidPlan("kv_heads do not fit in usize"))?;
    let head_dim = usize::try_from(head_dim)
        .map_err(|_| ExecutionError::InvalidPlan("head_dim does not fit in usize"))?;
    let max_seq = usize::try_from(max_seq)
        .map_err(|_| ExecutionError::InvalidPlan("max_seq does not fit in usize"))?;
    let buffer = bindings
        .get(value)
        .ok_or(ExecutionError::MissingBinding(value))?;
    if buffer == SCRATCHPAD {
        return Err(ExecutionError::InvalidPlan(
            "BufferId(0) is reserved for the scratchpad",
        ));
    }
    let data = buffers
        .get(buffer)
        .ok_or(ExecutionError::MissingBuffer(buffer))?;
    let expected = [batch, layers, 2, max_seq, kv_heads, head_dim]
        .iter()
        .try_fold(1_usize, |product, &extent| {
            product
                .checked_mul(extent)
                .ok_or(ExecutionError::Kernel(KernelError::ShapeOverflow))
        })?;
    expect_buffer_len(data.len(), expected)?;

    Ok(StateLayout {
        buffer,
        batch,
        layers,
        kv_heads,
        head_dim,
        max_seq,
    })
}

fn split_scratch<'a>(
    data: &'a mut [f32],
    ports: &[PortView; MAX_PORTS],
    port_count: usize,
) -> Result<[Option<&'a mut [f32]>; MAX_PORTS], ExecutionError> {
    let mut pieces: [Option<&'a mut [f32]>; MAX_PORTS] = core::array::from_fn(|_| None);
    let mut remaining = 0_usize;
    for port in ports.iter().take(port_count) {
        if matches!(port.source, Source::Scratch { .. }) {
            remaining += 1;
        }
    }

    let data_len = data.len();
    let mut rest: &'a mut [f32] = data;
    // Peel empty slices first. A zero-sized `Temp` is legal for an empty
    // tensor, and `split_at_mut(0)` yields a well-typed empty piece without
    // moving the cursor.
    for (index, port) in ports.iter().enumerate().take(port_count) {
        if let Source::Scratch { len: 0, .. } = port.source {
            let (piece, tail) = rest.split_at_mut(0);
            pieces[index] = Some(piece);
            rest = tail;
            remaining -= 1;
        }
    }

    let mut cursor = 0_usize;
    while remaining > 0 {
        let mut selected = None;
        let mut selected_start = usize::MAX;
        for (index, port) in ports.iter().enumerate().take(port_count) {
            if pieces[index].is_some() {
                continue;
            }
            if let Source::Scratch { start, .. } = port.source {
                if start < cursor {
                    return Err(ExecutionError::InvalidPlan(
                        "scratchpad slots of one command overlap",
                    ));
                }
                if start < selected_start {
                    selected_start = start;
                    selected = Some(index);
                }
            }
        }
        let Some(index) = selected else {
            return Err(ExecutionError::InvalidPlan(
                "scratchpad slot is missing from the command",
            ));
        };
        let Source::Scratch { start, len } = ports[index].source else {
            return Err(ExecutionError::InvalidPlan(
                "scratchpad slot is missing from the command",
            ));
        };
        let end = start
            .checked_add(len)
            .ok_or(ExecutionError::InvalidPlan("slot range overflows"))?;
        if end > data_len {
            return Err(ExecutionError::InvalidPlan(
                "slot exceeds scratchpad buffer",
            ));
        }
        let (_, tail) = rest.split_at_mut(start - cursor);
        let (piece, tail) = tail.split_at_mut(len);
        pieces[index] = Some(piece);
        rest = tail;
        cursor = end;
        remaining -= 1;
    }

    Ok(pieces)
}

fn validate_u32_payload(values: &[f32]) -> Result<(), ExecutionError> {
    // Values above this bound do not fit in `u32`. `2^32` is exactly
    // representable in `f32`, unlike `u32::MAX`.
    const U32_LIMIT: f32 = 4_294_967_296.0;
    for &value in values {
        if !(0.0..U32_LIMIT).contains(&value) || value.fract() != 0.0 {
            return Err(invalid_shape(
                "U32 tensor payload must contain non-negative integers",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference as kernels;
    use sl_ir::{CommandId, Dim, MemoryPlan, Shape, Slot, StateDecl, StateKind, IR_VERSION};

    const ALIGNMENT: u64 = 64;

    fn f32_value(shape: &[u32], role: ValueRole) -> Value {
        Value::new(DType::F32, Shape::consts(shape), role)
    }

    fn u32_value(shape: &[u32], role: ValueRole) -> Value {
        Value::new(DType::U32, Shape::consts(shape), role)
    }

    fn scalar_value() -> Value {
        Value::new(DType::U32, Shape::default(), ValueRole::Scalar)
    }

    fn command(op: Op, inputs: &[u32], outputs: &[u32], deps: &[u32]) -> Command {
        Command {
            op,
            inputs: inputs.iter().map(|&id| ValueId(id)).collect(),
            outputs: outputs.iter().map(|&id| ValueId(id)).collect(),
            deps: deps.iter().map(|&id| CommandId(id)).collect(),
            effect: Effect::None,
        }
    }

    fn stateful_command(
        op: Op,
        inputs: &[u32],
        outputs: &[u32],
        state: u32,
        deps: &[u32],
    ) -> Command {
        let mut command = command(op, inputs, outputs, deps);
        command.effect = Effect::WriteState(ValueId(state));
        command
    }

    fn bindings(pairs: &[(u32, u32)]) -> BindingTable {
        let mut table = BindingTable::new();
        for &(value, buffer) in pairs {
            table.insert(ValueId(value), BufferId(buffer));
        }
        table
    }

    fn scalars(pairs: &[(u32, u32)]) -> ScalarStore {
        let mut store = ScalarStore::new();
        for &(value, scalar) in pairs {
            store.insert(ValueId(value), scalar);
        }
        store
    }

    fn command_buffer(
        values: Vec<Value>,
        commands: Vec<Command>,
        plan: MemoryPlan,
        inputs: &[u32],
        outputs: &[u32],
        states: Vec<StateDecl>,
    ) -> CommandBuffer {
        CommandBuffer {
            ir_version: IR_VERSION,
            values,
            commands,
            plan,
            inputs: inputs.iter().map(|&id| ValueId(id)).collect(),
            outputs: outputs.iter().map(|&id| ValueId(id)).collect(),
            states,
        }
    }

    fn validate(ir: &CommandBuffer) {
        ir.validate(ALIGNMENT)
            .expect("test command buffer should satisfy sl-ir validation");
    }

    /// One-command graph: RmsNorm([1, 2]) -> Output([1, 2]).
    fn rms_norm_only_ir() -> CommandBuffer {
        let values = vec![
            f32_value(&[1, 2], ValueRole::Input),
            f32_value(&[2], ValueRole::Param),
            f32_value(&[1, 2], ValueRole::Output),
        ];
        let commands = vec![command(Op::RmsNorm { eps: 1e-5 }, &[0, 1], &[2], &[])];
        let ir = command_buffer(
            values,
            commands,
            MemoryPlan::default(),
            &[0],
            &[2],
            Vec::new(),
        );
        validate(&ir);
        ir
    }

    /// One-command graph with a `Temp`: RmsNorm -> Silu.
    fn temp_graph_ir(slot_bytes: u64, scratchpad_bytes: u64) -> CommandBuffer {
        let values = vec![
            f32_value(&[1, 2], ValueRole::Input),
            f32_value(&[2], ValueRole::Param),
            f32_value(&[1, 2], ValueRole::Temp),
            f32_value(&[1, 2], ValueRole::Output),
        ];
        let commands = vec![
            command(Op::RmsNorm { eps: 1e-5 }, &[0, 1], &[2], &[]),
            command(Op::Silu, &[2], &[3], &[0]),
        ];
        let plan = MemoryPlan {
            scratchpad_bytes,
            slots: vec![Slot {
                value: ValueId(2),
                offset: 0,
                bytes: slot_bytes,
            }],
        };
        command_buffer(values, commands, plan, &[0], &[3], Vec::new())
    }

    #[test]
    fn executes_minimal_rms_norm_matmul_add_graph() {
        let eps = 1e-5_f32;
        let values = vec![
            f32_value(&[2, 4], ValueRole::Input),
            f32_value(&[4], ValueRole::Param),
            f32_value(&[2, 4], ValueRole::Temp),
            f32_value(&[3, 4], ValueRole::Param),
            f32_value(&[2, 3], ValueRole::Temp),
            f32_value(&[2, 3], ValueRole::Input),
            f32_value(&[2, 3], ValueRole::Output),
        ];
        let commands = vec![
            command(Op::RmsNorm { eps }, &[0, 1], &[2], &[]),
            command(Op::MatMulNT, &[2, 3], &[4], &[0]),
            command(Op::Add, &[4, 5], &[6], &[1]),
        ];
        let plan = MemoryPlan {
            scratchpad_bytes: 128,
            slots: vec![
                Slot {
                    value: ValueId(2),
                    offset: 0,
                    bytes: 32,
                },
                Slot {
                    value: ValueId(4),
                    offset: 64,
                    bytes: 24,
                },
            ],
        };
        let ir = command_buffer(values, commands, plan, &[0, 5], &[6], Vec::new());
        validate(&ir);

        let x: Vec<f32> = (0..8).map(|index| (index as f32 - 3.0) * 0.25).collect();
        let weight = vec![1.0, 0.5, 1.5, 0.75];
        let w: Vec<f32> = (0..12)
            .map(|index| ((index % 5) as f32 - 2.0) * 0.1)
            .collect();
        let residual = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6];

        let mut expected_rms = vec![0.0_f32; 8];
        kernels::rms_norm(&x, &weight, &mut expected_rms, 2, 4, eps)
            .expect("direct rms_norm should succeed");
        let mut expected_matmul = vec![0.0_f32; 6];
        kernels::matmul_nt(&expected_rms, &w, &mut expected_matmul, 1, 2, 4, 3)
            .expect("direct matmul_nt should succeed");
        let mut expected = vec![0.0_f32; 6];
        kernels::add(&expected_matmul, &residual, &mut expected, 6)
            .expect("direct add should succeed");

        let bindings = bindings(&[(0, 1), (1, 2), (3, 3), (5, 4), (6, 5)]);
        let mut buffers = BufferStore::new();
        buffers.insert(BufferId(1), x);
        buffers.insert(BufferId(2), weight);
        buffers.insert(BufferId(3), w);
        buffers.insert(BufferId(4), residual);
        buffers.insert(BufferId(5), vec![0.0; 6]);
        buffers.insert(SCRATCHPAD, vec![0.0; 32]);

        execute(&ir, &bindings, &mut buffers, &ScalarStore::new())
            .expect("minimal graph should execute");

        assert_eq!(buffers.get(BufferId(5)).unwrap(), expected.as_slice());
        assert!(buffers.get(SCRATCHPAD).is_some());
    }

    #[test]
    fn matmul_nt_broadcasts_a_single_weight_across_batches() {
        let values = vec![
            f32_value(&[2, 3, 4], ValueRole::Input),
            f32_value(&[5, 4], ValueRole::Param),
            f32_value(&[2, 3, 5], ValueRole::Output),
        ];
        let commands = vec![command(Op::MatMulNT, &[0, 1], &[2], &[])];
        let ir = command_buffer(
            values,
            commands,
            MemoryPlan::default(),
            &[0],
            &[2],
            Vec::new(),
        );
        validate(&ir);

        let a: Vec<f32> = (0..24).map(|index| (index as f32 - 7.0) * 0.1).collect();
        let b: Vec<f32> = (0..20).map(|index| (index as f32 - 5.0) * 0.05).collect();
        let mut expected = vec![0.0_f32; 30];
        for batch in 0..2 {
            kernels::matmul_nt(
                &a[batch * 12..(batch + 1) * 12],
                &b,
                &mut expected[batch * 15..(batch + 1) * 15],
                1,
                3,
                4,
                5,
            )
            .expect("direct matmul should succeed");
        }

        let mut buffers = BufferStore::new();
        buffers.insert(BufferId(1), a);
        buffers.insert(BufferId(2), b);
        buffers.insert(BufferId(3), vec![0.0; 30]);
        execute(
            &ir,
            &bindings(&[(0, 1), (1, 2), (2, 3)]),
            &mut buffers,
            &ScalarStore::new(),
        )
        .expect("broadcast matmul should execute");

        assert_eq!(buffers.get(BufferId(3)).unwrap(), expected.as_slice());
    }

    #[test]
    fn matmul_nn_broadcasts_a_single_activation_across_batches() {
        let values = vec![
            f32_value(&[3, 4], ValueRole::Input),
            f32_value(&[2, 4, 5], ValueRole::Param),
            f32_value(&[2, 3, 5], ValueRole::Output),
        ];
        let commands = vec![command(Op::MatMulNN, &[0, 1], &[2], &[])];
        let ir = command_buffer(
            values,
            commands,
            MemoryPlan::default(),
            &[0],
            &[2],
            Vec::new(),
        );
        validate(&ir);

        let a: Vec<f32> = (0..12).map(|index| (index as f32 - 5.0) * 0.2).collect();
        let b: Vec<f32> = (0..40).map(|index| (index as f32 - 9.0) * 0.03).collect();
        let mut expected = vec![0.0_f32; 30];
        for batch in 0..2 {
            kernels::matmul_nn(
                &a,
                &b[batch * 20..(batch + 1) * 20],
                &mut expected[batch * 15..(batch + 1) * 15],
                1,
                3,
                4,
                5,
            )
            .expect("direct matmul should succeed");
        }

        let mut buffers = BufferStore::new();
        buffers.insert(BufferId(1), a);
        buffers.insert(BufferId(2), b);
        buffers.insert(BufferId(3), vec![0.0; 30]);
        execute(
            &ir,
            &bindings(&[(0, 1), (1, 2), (2, 3)]),
            &mut buffers,
            &ScalarStore::new(),
        )
        .expect("broadcast matmul should execute");

        assert_eq!(buffers.get(BufferId(3)).unwrap(), expected.as_slice());
    }

    #[test]
    fn matmul_rejects_incompatible_batches() {
        let values = vec![
            f32_value(&[2, 3, 4], ValueRole::Input),
            f32_value(&[3, 5, 4], ValueRole::Param),
            f32_value(&[2, 3, 5], ValueRole::Output),
        ];
        let commands = vec![command(Op::MatMulNT, &[0, 1], &[2], &[])];
        let ir = command_buffer(
            values,
            commands,
            MemoryPlan::default(),
            &[0],
            &[2],
            Vec::new(),
        );
        validate(&ir);

        let mut buffers = BufferStore::new();
        buffers.insert(BufferId(1), vec![0.0; 24]);
        buffers.insert(BufferId(2), vec![0.0; 60]);
        buffers.insert(BufferId(3), vec![0.0; 30]);
        let error = execute(
            &ir,
            &bindings(&[(0, 1), (1, 2), (2, 3)]),
            &mut buffers,
            &ScalarStore::new(),
        )
        .expect_err("incompatible batches must be rejected");

        assert_eq!(
            error,
            ExecutionError::Kernel(KernelError::InvalidShape(
                "matmul batch dimensions are not broadcastable"
            ))
        );
    }

    fn attention_ir(seq: u32) -> CommandBuffer {
        let values = vec![
            f32_value(&[1, 1, seq, 2], ValueRole::Input),
            f32_value(&[1, 1, seq, 2], ValueRole::Input),
            f32_value(&[1, 1, seq, 2], ValueRole::Input),
            scalar_value(),
            f32_value(&[1, 1, seq, 2], ValueRole::Output),
            f32_value(&[1, 1, 2, 4, 1, 2], ValueRole::State),
        ];
        let commands = vec![stateful_command(
            Op::CausalAttention {
                layer: 0,
                heads: 1,
                kv_heads: 1,
                head_dim: 2,
                scale: 0.5,
            },
            &[0, 1, 2, 3],
            &[4],
            5,
            &[],
        )];
        let states = vec![StateDecl {
            value: ValueId(5),
            kind: StateKind::KvCache {
                batch: 1,
                layers: 1,
                kv_heads: 1,
                head_dim: 2,
                max_seq: Dim::Const(4),
                dtype: DType::F32,
            },
        }];
        let ir = command_buffer(
            values,
            commands,
            MemoryPlan::default(),
            &[0, 1, 2],
            &[4],
            states,
        );
        validate(&ir);
        ir
    }

    #[test]
    fn executes_causal_attention_prefill_and_decode() {
        let scale = 0.5_f32;
        let q_prefill = vec![0.1, -0.2, 0.3, 0.4];
        let k_prefill = vec![0.5, -0.1, 0.2, 0.7];
        let v_prefill = vec![0.9, -0.4, 0.6, 0.2];
        let q_decode = vec![0.3, 0.8];
        let k_decode = vec![-0.2, 0.4];
        let v_decode = vec![0.5, -0.7];

        let mut expected_cache = vec![0.0_f32; 16];
        let mut expected_prefill = vec![0.0_f32; 4];
        kernels::causal_attention(
            &q_prefill,
            &k_prefill,
            &v_prefill,
            &mut expected_cache,
            &mut expected_prefill,
            1,
            1,
            0,
            1,
            1,
            2,
            0,
            2,
            4,
            scale,
        )
        .expect("direct prefill should succeed");
        let prefill_ir = attention_ir(2);
        let prefill_bindings = bindings(&[(0, 1), (1, 2), (2, 3), (4, 4), (5, 5)]);
        let prefill_scalars = scalars(&[(3, 0)]);
        let mut buffers = BufferStore::new();
        buffers.insert(BufferId(1), q_prefill);
        buffers.insert(BufferId(2), k_prefill);
        buffers.insert(BufferId(3), v_prefill);
        buffers.insert(BufferId(4), vec![0.0; 4]);
        buffers.insert(BufferId(5), vec![0.0; 16]);
        execute(
            &prefill_ir,
            &prefill_bindings,
            &mut buffers,
            &prefill_scalars,
        )
        .expect("prefill should execute");
        assert_eq!(
            buffers.get(BufferId(4)).unwrap(),
            expected_prefill.as_slice()
        );
        assert_eq!(buffers.get(BufferId(5)).unwrap(), expected_cache.as_slice());

        let mut expected_decode = vec![0.0_f32; 2];
        kernels::causal_attention(
            &q_decode,
            &k_decode,
            &v_decode,
            &mut expected_cache,
            &mut expected_decode,
            1,
            1,
            0,
            1,
            1,
            1,
            2,
            2,
            4,
            scale,
        )
        .expect("direct decode should succeed");

        let decode_ir = attention_ir(1);
        let decode_bindings = bindings(&[(0, 6), (1, 7), (2, 8), (4, 9), (5, 5)]);
        let decode_scalars = scalars(&[(3, 2)]);
        buffers.insert(BufferId(6), q_decode);
        buffers.insert(BufferId(7), k_decode);
        buffers.insert(BufferId(8), v_decode);
        buffers.insert(BufferId(9), vec![0.0; 2]);
        execute(&decode_ir, &decode_bindings, &mut buffers, &decode_scalars)
            .expect("decode should execute");
        assert_eq!(
            buffers.get(BufferId(9)).unwrap(),
            expected_decode.as_slice()
        );
        assert_eq!(buffers.get(BufferId(5)).unwrap(), expected_cache.as_slice());
    }

    #[test]
    fn reuses_scratchpad_slots_with_disjoint_lifetimes() {
        let eps = 1e-5_f32;
        let values = vec![
            f32_value(&[1, 4], ValueRole::Input),
            f32_value(&[4], ValueRole::Param),
            f32_value(&[1, 4], ValueRole::Temp),
            f32_value(&[1, 4], ValueRole::Input),
            f32_value(&[1, 4], ValueRole::Output),
            f32_value(&[1, 4], ValueRole::Input),
            f32_value(&[4], ValueRole::Param),
            f32_value(&[1, 4], ValueRole::Temp),
            f32_value(&[1, 4], ValueRole::Input),
            f32_value(&[1, 4], ValueRole::Output),
        ];
        let commands = vec![
            command(Op::RmsNorm { eps }, &[0, 1], &[2], &[]),
            command(Op::Add, &[2, 3], &[4], &[0]),
            command(Op::RmsNorm { eps }, &[5, 6], &[7], &[1]),
            command(Op::Add, &[7, 8], &[9], &[2]),
        ];
        let plan = MemoryPlan {
            scratchpad_bytes: 64,
            slots: vec![
                Slot {
                    value: ValueId(2),
                    offset: 0,
                    bytes: 16,
                },
                Slot {
                    value: ValueId(7),
                    offset: 0,
                    bytes: 16,
                },
            ],
        };
        let ir = command_buffer(values, commands, plan, &[0, 3, 5, 8], &[4, 9], Vec::new());
        // The plan is only valid because the two Temps have disjoint live
        // ranges; sl-ir's own validation enforces that invariant.
        validate(&ir);

        let x0 = vec![0.2, -0.4, 0.6, 0.8];
        let w0 = vec![1.0, 0.5, 1.5, 0.25];
        let r0 = vec![0.1, 0.2, -0.1, -0.2];
        let x1 = vec![-0.3, 0.7, 0.1, -0.5];
        let w1 = vec![0.75, 1.25, 0.5, 1.0];
        let r1 = vec![-0.4, 0.3, 0.2, -0.1];

        let mut t0 = vec![0.0_f32; 4];
        kernels::rms_norm(&x0, &w0, &mut t0, 1, 4, eps).unwrap();
        let mut out0 = vec![0.0_f32; 4];
        kernels::add(&t0, &r0, &mut out0, 4).unwrap();
        let mut t1 = vec![0.0_f32; 4];
        kernels::rms_norm(&x1, &w1, &mut t1, 1, 4, eps).unwrap();
        let mut out1 = vec![0.0_f32; 4];
        kernels::add(&t1, &r1, &mut out1, 4).unwrap();

        let bindings = bindings(&[
            (0, 1),
            (1, 2),
            (3, 3),
            (4, 4),
            (5, 5),
            (6, 6),
            (8, 7),
            (9, 8),
        ]);
        let mut buffers = BufferStore::new();
        buffers.insert(BufferId(1), x0);
        buffers.insert(BufferId(2), w0);
        buffers.insert(BufferId(3), r0);
        buffers.insert(BufferId(4), vec![0.0; 4]);
        buffers.insert(BufferId(5), x1);
        buffers.insert(BufferId(6), w1);
        buffers.insert(BufferId(7), r1);
        buffers.insert(BufferId(8), vec![0.0; 4]);
        buffers.insert(SCRATCHPAD, vec![0.0; 16]);

        execute(&ir, &bindings, &mut buffers, &ScalarStore::new())
            .expect("scratchpad reuse should execute");
        assert_eq!(buffers.get(BufferId(4)).unwrap(), out0.as_slice());
        assert_eq!(buffers.get(BufferId(8)).unwrap(), out1.as_slice());
    }

    #[test]
    fn reports_missing_binding() {
        let ir = rms_norm_only_ir();
        let bindings = bindings(&[(0, 1), (2, 3)]);
        let mut buffers = BufferStore::new();
        buffers.insert(BufferId(1), vec![1.0, 2.0]);
        buffers.insert(BufferId(3), vec![0.0; 2]);
        let error = execute(&ir, &bindings, &mut buffers, &ScalarStore::new())
            .expect_err("missing binding must be rejected");
        assert_eq!(error, ExecutionError::MissingBinding(ValueId(1)));
    }

    #[test]
    fn reports_missing_buffer() {
        let ir = rms_norm_only_ir();
        let bindings = bindings(&[(0, 1), (1, 2), (2, 3)]);
        let mut buffers = BufferStore::new();
        buffers.insert(BufferId(1), vec![1.0, 2.0]);
        buffers.insert(BufferId(3), vec![0.0; 2]);
        let error = execute(&ir, &bindings, &mut buffers, &ScalarStore::new())
            .expect_err("missing buffer must be rejected");
        assert_eq!(error, ExecutionError::MissingBuffer(BufferId(2)));
    }

    #[test]
    fn reports_missing_scalar() {
        let values = vec![
            f32_value(&[1, 1, 2], ValueRole::Input),
            scalar_value(),
            f32_value(&[1, 2], ValueRole::Output),
        ];
        let commands = vec![command(Op::LastToken, &[0, 1], &[2], &[])];
        let ir = command_buffer(
            values,
            commands,
            MemoryPlan::default(),
            &[0],
            &[2],
            Vec::new(),
        );
        validate(&ir);
        let bindings = bindings(&[(0, 1), (2, 2)]);
        let mut buffers = BufferStore::new();
        buffers.insert(BufferId(1), vec![1.0, 2.0]);
        buffers.insert(BufferId(2), vec![0.0; 2]);
        let error = execute(&ir, &bindings, &mut buffers, &ScalarStore::new())
            .expect_err("missing scalar must be rejected");
        assert_eq!(error, ExecutionError::MissingScalar(ValueId(1)));
    }

    #[test]
    fn reports_plan_out_of_bounds() {
        let ir = temp_graph_ir(8, 4);
        assert!(ir.validate(ALIGNMENT).is_err());
        let bindings = bindings(&[(0, 1), (1, 2), (3, 4)]);
        let mut buffers = BufferStore::new();
        buffers.insert(BufferId(1), vec![1.0, 2.0]);
        buffers.insert(BufferId(2), vec![1.0, 1.0]);
        buffers.insert(BufferId(4), vec![0.0; 2]);
        buffers.insert(SCRATCHPAD, vec![0.0; 1]);
        let error = execute(&ir, &bindings, &mut buffers, &ScalarStore::new())
            .expect_err("out-of-bounds slot must be rejected");
        assert_eq!(
            error,
            ExecutionError::InvalidPlan("slot exceeds scratchpad")
        );
    }

    #[test]
    fn reports_slot_size_mismatch() {
        let ir = temp_graph_ir(12, 64);
        assert!(ir.validate(ALIGNMENT).is_err());
        let bindings = bindings(&[(0, 1), (1, 2), (3, 4)]);
        let mut buffers = BufferStore::new();
        buffers.insert(BufferId(1), vec![1.0, 2.0]);
        buffers.insert(BufferId(2), vec![1.0, 1.0]);
        buffers.insert(BufferId(4), vec![0.0; 2]);
        buffers.insert(SCRATCHPAD, vec![0.0; 16]);
        let error = execute(&ir, &bindings, &mut buffers, &ScalarStore::new())
            .expect_err("slot size mismatch must be rejected");
        assert_eq!(
            error,
            ExecutionError::InvalidPlan("slot size does not match the value size")
        );
    }

    #[test]
    fn reports_missing_scratchpad() {
        let ir = temp_graph_ir(8, 64);
        validate(&ir);
        let bindings = bindings(&[(0, 1), (1, 2), (3, 4)]);
        let mut buffers = BufferStore::new();
        buffers.insert(BufferId(1), vec![1.0, 2.0]);
        buffers.insert(BufferId(2), vec![1.0, 1.0]);
        buffers.insert(BufferId(4), vec![0.0; 2]);
        let error = execute(&ir, &bindings, &mut buffers, &ScalarStore::new())
            .expect_err("missing scratchpad must be rejected");
        assert_eq!(error, ExecutionError::MissingBuffer(SCRATCHPAD));
    }

    #[test]
    fn reports_buffer_length_mismatch() {
        let ir = rms_norm_only_ir();
        let bindings = bindings(&[(0, 1), (1, 2), (2, 3)]);
        let mut buffers = BufferStore::new();
        buffers.insert(BufferId(1), vec![1.0, 2.0]);
        buffers.insert(BufferId(2), vec![1.0]);
        buffers.insert(BufferId(3), vec![0.0; 2]);
        let error = execute(&ir, &bindings, &mut buffers, &ScalarStore::new())
            .expect_err("wrong buffer length must be rejected");
        assert_eq!(
            error,
            ExecutionError::Kernel(KernelError::LengthMismatch {
                expected: 2,
                actual: 1,
            })
        );
    }

    #[test]
    fn executes_zero_row_graph_without_panicking() {
        let values = vec![
            f32_value(&[0, 4], ValueRole::Input),
            f32_value(&[4], ValueRole::Param),
            f32_value(&[3, 4], ValueRole::Param),
            f32_value(&[0, 3], ValueRole::Output),
        ];
        let commands = vec![command(
            Op::RmsNormMatMul { eps: 1e-5 },
            &[0, 1, 2],
            &[3],
            &[],
        )];
        let ir = command_buffer(
            values,
            commands,
            MemoryPlan::default(),
            &[0],
            &[3],
            Vec::new(),
        );
        validate(&ir);

        let bindings = bindings(&[(0, 1), (1, 2), (2, 3), (3, 4)]);
        let mut buffers = BufferStore::new();
        buffers.insert(BufferId(1), Vec::new());
        buffers.insert(BufferId(2), vec![1.0; 4]);
        buffers.insert(BufferId(3), vec![1.0; 12]);
        buffers.insert(BufferId(4), Vec::new());
        execute(&ir, &bindings, &mut buffers, &ScalarStore::new())
            .expect("rows == 0 should execute without panicking");
        assert!(buffers.get(BufferId(4)).unwrap().is_empty());
    }

    #[test]
    fn executes_matmul_nn() {
        let values = vec![
            f32_value(&[2, 3], ValueRole::Input),
            f32_value(&[3, 4], ValueRole::Param),
            f32_value(&[2, 4], ValueRole::Output),
        ];
        let commands = vec![command(Op::MatMulNN, &[0, 1], &[2], &[])];
        let ir = command_buffer(
            values,
            commands,
            MemoryPlan::default(),
            &[0],
            &[2],
            Vec::new(),
        );
        validate(&ir);

        let a = vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.6];
        let b: Vec<f32> = (0..12).map(|index| (index as f32 - 5.0) * 0.1).collect();
        let mut expected = vec![0.0_f32; 8];
        kernels::matmul_nn(&a, &b, &mut expected, 1, 2, 3, 4)
            .expect("direct matmul_nn should succeed");

        let bindings = bindings(&[(0, 1), (1, 2), (2, 3)]);
        let mut buffers = BufferStore::new();
        buffers.insert(BufferId(1), a);
        buffers.insert(BufferId(2), b);
        buffers.insert(BufferId(3), vec![0.0; 8]);
        execute(&ir, &bindings, &mut buffers, &ScalarStore::new())
            .expect("matmul_nn should execute");
        assert_eq!(buffers.get(BufferId(3)).unwrap(), expected.as_slice());
    }

    #[test]
    fn executes_elementwise_and_softmax_chain() {
        let values = vec![
            f32_value(&[2, 3], ValueRole::Input),
            f32_value(&[2, 3], ValueRole::Input),
            f32_value(&[2, 3], ValueRole::Temp),
            f32_value(&[2, 3], ValueRole::Temp),
            f32_value(&[2, 3], ValueRole::Temp),
            f32_value(&[2, 3], ValueRole::Temp),
            f32_value(&[2, 3], ValueRole::Output),
        ];
        let commands = vec![
            command(Op::Silu, &[0], &[2], &[]),
            command(Op::Mul, &[2, 1], &[3], &[0]),
            command(Op::Add, &[3, 0], &[4], &[1]),
            command(Op::Scale { factor: 1.5 }, &[4], &[5], &[2]),
            command(Op::Softmax, &[5], &[6], &[3]),
        ];
        let plan = MemoryPlan {
            scratchpad_bytes: 256,
            slots: vec![
                Slot {
                    value: ValueId(2),
                    offset: 0,
                    bytes: 24,
                },
                Slot {
                    value: ValueId(3),
                    offset: 64,
                    bytes: 24,
                },
                Slot {
                    value: ValueId(4),
                    offset: 128,
                    bytes: 24,
                },
                Slot {
                    value: ValueId(5),
                    offset: 192,
                    bytes: 24,
                },
            ],
        };
        let ir = command_buffer(values, commands, plan, &[0, 1], &[6], Vec::new());
        validate(&ir);

        let a = vec![0.5, -1.0, 2.0, -0.25, 0.75, 1.5];
        let b = vec![1.0, 0.5, -0.5, 2.0, -1.5, 0.25];
        let mut silu = a.clone();
        kernels::silu(&mut silu).expect("direct silu should succeed");
        let mut mul = vec![0.0_f32; 6];
        kernels::mul(&silu, &b, &mut mul, 6).expect("direct mul should succeed");
        let mut add = vec![0.0_f32; 6];
        kernels::add(&mul, &a, &mut add, 6).expect("direct add should succeed");
        let mut scale = add.clone();
        kernels::scale(&mut scale, 1.5).expect("direct scale should succeed");
        let mut expected = scale.clone();
        kernels::softmax(&mut expected, 2, 3).expect("direct softmax should succeed");

        let bindings = bindings(&[(0, 1), (1, 2), (6, 3)]);
        let mut buffers = BufferStore::new();
        buffers.insert(BufferId(1), a);
        buffers.insert(BufferId(2), b);
        buffers.insert(BufferId(3), vec![0.0; 6]);
        buffers.insert(SCRATCHPAD, vec![0.0; 64]);
        execute(&ir, &bindings, &mut buffers, &ScalarStore::new())
            .expect("elementwise chain should execute");
        assert_eq!(buffers.get(BufferId(3)).unwrap(), expected.as_slice());
    }

    #[test]
    fn executes_softmax_causal() {
        let values = vec![
            f32_value(&[2, 3], ValueRole::Input),
            scalar_value(),
            scalar_value(),
            f32_value(&[2, 3], ValueRole::Output),
        ];
        let commands = vec![command(Op::SoftmaxCausal, &[0, 1, 2], &[3], &[])];
        let ir = command_buffer(
            values,
            commands,
            MemoryPlan::default(),
            &[0],
            &[3],
            Vec::new(),
        );
        validate(&ir);

        let x = vec![0.5, -0.5, 1.0, -1.0, 0.25, 2.0];
        let mut expected = x.clone();
        kernels::softmax_causal(&mut expected, 2, 3, 2, 1)
            .expect("direct softmax_causal should succeed");

        let bindings = bindings(&[(0, 1), (3, 2)]);
        let mut buffers = BufferStore::new();
        buffers.insert(BufferId(1), x);
        buffers.insert(BufferId(2), vec![0.0; 6]);
        execute(&ir, &bindings, &mut buffers, &scalars(&[(1, 2), (2, 1)]))
            .expect("softmax_causal should execute");
        assert_eq!(buffers.get(BufferId(2)).unwrap(), expected.as_slice());
    }

    #[test]
    fn executes_rope_variants() {
        use sl_ir::RopeVariant;

        let x: Vec<f32> = (0..24).map(|index| (index as f32 - 11.0) * 0.05).collect();
        let positions = vec![0_u32, 1, 2];
        for variant in [RopeVariant::Neox, RopeVariant::GptJ] {
            let values = vec![
                f32_value(&[1, 2, 3, 4], ValueRole::Input),
                u32_value(&[1, 3], ValueRole::Input),
                f32_value(&[1, 2, 3, 4], ValueRole::Output),
            ];
            let commands = vec![command(
                Op::Rope {
                    variant,
                    theta: 10_000.0,
                },
                &[0, 1],
                &[2],
                &[],
            )];
            let ir = command_buffer(
                values,
                commands,
                MemoryPlan::default(),
                &[0, 1],
                &[2],
                Vec::new(),
            );
            validate(&ir);

            let mut expected = x.clone();
            match variant {
                RopeVariant::Neox => {
                    kernels::rope_neox(&mut expected, &positions, 1, 2, 3, 4, 10_000.0)
                        .expect("direct rope_neox should succeed");
                }
                RopeVariant::GptJ => {
                    kernels::rope_gptj(&mut expected, &positions, 1, 2, 3, 4, 10_000.0)
                        .expect("direct rope_gptj should succeed");
                }
            }

            let bindings = bindings(&[(0, 1), (1, 2), (2, 3)]);
            let mut buffers = BufferStore::new();
            buffers.insert(BufferId(1), x.clone());
            buffers.insert_u32(BufferId(2), positions.clone());
            buffers.insert(BufferId(3), vec![0.0; 24]);
            execute(&ir, &bindings, &mut buffers, &ScalarStore::new())
                .expect("rope should execute");
            assert_eq!(buffers.get(BufferId(3)).unwrap(), expected.as_slice());
        }
    }

    fn gather_case(exact_u32: bool) {
        let values = vec![
            f32_value(&[5, 3], ValueRole::Param),
            u32_value(&[4], ValueRole::Input),
            f32_value(&[4, 3], ValueRole::Output),
        ];
        let commands = vec![command(Op::Gather, &[0, 1], &[2], &[])];
        let ir = command_buffer(
            values,
            commands,
            MemoryPlan::default(),
            &[1],
            &[2],
            Vec::new(),
        );
        validate(&ir);

        let table: Vec<f32> = (0..15).map(|index| (index as f32 - 7.0) * 0.1).collect();
        let ids = vec![0_u32, 4, 2, 2];
        let mut expected = vec![0.0_f32; 12];
        kernels::gather(&table, &ids, &mut expected, 5, 3, 4)
            .expect("direct gather should succeed");

        let bindings = bindings(&[(0, 1), (1, 2), (2, 3)]);
        let mut buffers = BufferStore::new();
        buffers.insert(BufferId(1), table);
        if exact_u32 {
            buffers.insert_u32(BufferId(2), ids);
        } else {
            buffers.insert(BufferId(2), ids.iter().map(|&id| id as f32).collect());
        }
        buffers.insert(BufferId(3), vec![0.0; 12]);
        execute(&ir, &bindings, &mut buffers, &ScalarStore::new()).expect("gather should execute");
        assert_eq!(buffers.get(BufferId(3)).unwrap(), expected.as_slice());
    }

    #[test]
    fn executes_gather_with_exact_u32_ids() {
        gather_case(true);
    }

    #[test]
    fn executes_gather_with_numeric_f32_ids() {
        gather_case(false);
    }

    #[test]
    fn executes_last_token() {
        let values = vec![
            f32_value(&[2, 3, 4], ValueRole::Input),
            scalar_value(),
            f32_value(&[2, 4], ValueRole::Output),
        ];
        let commands = vec![command(Op::LastToken, &[0, 1], &[2], &[])];
        let ir = command_buffer(
            values,
            commands,
            MemoryPlan::default(),
            &[0],
            &[2],
            Vec::new(),
        );
        validate(&ir);

        let input: Vec<f32> = (0..24).map(|index| index as f32 * 0.25).collect();
        let mut expected = vec![0.0_f32; 8];
        kernels::last_token(&input, &mut expected, 2, 3, 4, 2)
            .expect("direct last_token should succeed");

        let bindings = bindings(&[(0, 1), (2, 2)]);
        let mut buffers = BufferStore::new();
        buffers.insert(BufferId(1), input);
        buffers.insert(BufferId(2), vec![0.0; 8]);
        execute(&ir, &bindings, &mut buffers, &scalars(&[(1, 2)]))
            .expect("last_token should execute");
        assert_eq!(buffers.get(BufferId(2)).unwrap(), expected.as_slice());
    }

    #[test]
    fn executes_split_and_merge_heads() {
        let values = vec![
            f32_value(&[2, 3, 4], ValueRole::Input),
            f32_value(&[2, 2, 3, 2], ValueRole::Temp),
            f32_value(&[2, 3, 4], ValueRole::Output),
        ];
        let commands = vec![
            command(
                Op::SplitHeads {
                    heads: 2,
                    head_dim: 2,
                },
                &[0],
                &[1],
                &[],
            ),
            command(
                Op::MergeHeads {
                    heads: 2,
                    head_dim: 2,
                },
                &[1],
                &[2],
                &[0],
            ),
        ];
        let plan = MemoryPlan {
            scratchpad_bytes: 128,
            slots: vec![Slot {
                value: ValueId(1),
                offset: 0,
                bytes: 96,
            }],
        };
        let ir = command_buffer(values, commands, plan, &[0], &[2], Vec::new());
        validate(&ir);

        let input: Vec<f32> = (0..24).map(|index| index as f32 * 0.5).collect();
        let mut split = vec![0.0_f32; 24];
        kernels::split_heads(&input, &mut split, 2, 3, 2, 2)
            .expect("direct split_heads should succeed");
        let mut expected = vec![0.0_f32; 24];
        kernels::merge_heads(&split, &mut expected, 2, 3, 2, 2)
            .expect("direct merge_heads should succeed");

        let bindings = bindings(&[(0, 1), (2, 2)]);
        let mut buffers = BufferStore::new();
        buffers.insert(BufferId(1), input);
        buffers.insert(BufferId(2), vec![0.0; 24]);
        buffers.insert(SCRATCHPAD, vec![0.0; 32]);
        execute(&ir, &bindings, &mut buffers, &ScalarStore::new())
            .expect("split/merge heads should execute");
        assert_eq!(buffers.get(BufferId(2)).unwrap(), expected.as_slice());
    }

    #[test]
    fn executes_kv_append() {
        let values = vec![
            f32_value(&[1, 1, 2, 2], ValueRole::Input),
            f32_value(&[1, 1, 2, 2], ValueRole::Input),
            scalar_value(),
            f32_value(&[1, 1, 2, 4, 1, 2], ValueRole::State),
        ];
        let commands = vec![stateful_command(
            Op::KvAppend { layer: 0 },
            &[0, 1, 2],
            &[],
            3,
            &[],
        )];
        let states = vec![StateDecl {
            value: ValueId(3),
            kind: StateKind::KvCache {
                batch: 1,
                layers: 1,
                kv_heads: 1,
                head_dim: 2,
                max_seq: Dim::Const(4),
                dtype: DType::F32,
            },
        }];
        let ir = command_buffer(
            values,
            commands,
            MemoryPlan::default(),
            &[0, 1],
            &[],
            states,
        );
        validate(&ir);

        let k = vec![0.5, -0.1, 0.2, 0.7];
        let v = vec![0.9, -0.4, 0.6, 0.2];
        let mut expected = vec![0.0_f32; 16];
        kernels::kv_append(&k, &v, &mut expected, 1, 1, 0, 1, 2, 4, 1, 2)
            .expect("direct kv_append should succeed");

        let bindings = bindings(&[(0, 1), (1, 2), (3, 3)]);
        let mut buffers = BufferStore::new();
        buffers.insert(BufferId(1), k);
        buffers.insert(BufferId(2), v);
        buffers.insert(BufferId(3), vec![0.0; 16]);
        execute(&ir, &bindings, &mut buffers, &scalars(&[(2, 1)]))
            .expect("kv_append should execute");
        assert_eq!(buffers.get(BufferId(3)).unwrap(), expected.as_slice());
    }

    #[test]
    fn executes_swiglu() {
        let values = vec![
            f32_value(&[2, 3], ValueRole::Input),
            f32_value(&[4, 3], ValueRole::Param),
            f32_value(&[4, 3], ValueRole::Param),
            f32_value(&[3, 4], ValueRole::Param),
            f32_value(&[2, 3], ValueRole::Output),
        ];
        let commands = vec![command(Op::SwiGlu, &[0, 1, 2, 3], &[4], &[])];
        let ir = command_buffer(
            values,
            commands,
            MemoryPlan::default(),
            &[0],
            &[4],
            Vec::new(),
        );
        validate(&ir);

        let x = vec![0.5, -1.0, 2.0, -0.25, 0.75, 1.5];
        let gate: Vec<f32> = (0..12).map(|index| (index as f32 - 5.0) * 0.1).collect();
        let up: Vec<f32> = (0..12).map(|index| (index as f32 - 3.0) * 0.15).collect();
        let down: Vec<f32> = (0..12).map(|index| (index as f32 - 6.0) * 0.05).collect();
        let mut expected = vec![0.0_f32; 6];
        kernels::swiglu(&x, &gate, &up, &down, &mut expected, 2, 3, 4)
            .expect("direct swiglu should succeed");

        let bindings = bindings(&[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]);
        let mut buffers = BufferStore::new();
        buffers.insert(BufferId(1), x);
        buffers.insert(BufferId(2), gate);
        buffers.insert(BufferId(3), up);
        buffers.insert(BufferId(4), down);
        buffers.insert(BufferId(5), vec![0.0; 6]);
        execute(&ir, &bindings, &mut buffers, &ScalarStore::new()).expect("swiglu should execute");
        assert_eq!(buffers.get(BufferId(5)).unwrap(), expected.as_slice());
    }

    #[test]
    fn executes_rms_norm_matmul() {
        let values = vec![
            f32_value(&[2, 3], ValueRole::Input),
            f32_value(&[3], ValueRole::Param),
            f32_value(&[4, 3], ValueRole::Param),
            f32_value(&[2, 4], ValueRole::Output),
        ];
        let commands = vec![command(
            Op::RmsNormMatMul { eps: 1e-6 },
            &[0, 1, 2],
            &[3],
            &[],
        )];
        let ir = command_buffer(
            values,
            commands,
            MemoryPlan::default(),
            &[0],
            &[3],
            Vec::new(),
        );
        validate(&ir);

        let x = vec![0.5, -1.0, 2.0, -0.25, 0.75, 1.5];
        let weight = vec![1.0, 0.5, 1.5];
        let matrix: Vec<f32> = (0..12).map(|index| (index as f32 - 5.0) * 0.1).collect();
        let mut expected = vec![0.0_f32; 8];
        kernels::rms_norm_matmul(&x, &weight, &matrix, &mut expected, 2, 3, 4, 1e-6)
            .expect("direct rms_norm_matmul should succeed");

        let bindings = bindings(&[(0, 1), (1, 2), (2, 3), (3, 4)]);
        let mut buffers = BufferStore::new();
        buffers.insert(BufferId(1), x);
        buffers.insert(BufferId(2), weight);
        buffers.insert(BufferId(3), matrix);
        buffers.insert(BufferId(4), vec![0.0; 8]);
        execute(&ir, &bindings, &mut buffers, &ScalarStore::new())
            .expect("rms_norm_matmul should execute");
        assert_eq!(buffers.get(BufferId(4)).unwrap(), expected.as_slice());
    }

    #[test]
    fn executes_rank_one_rms_norm() {
        let values = vec![
            f32_value(&[4], ValueRole::Input),
            f32_value(&[4], ValueRole::Param),
            f32_value(&[4], ValueRole::Output),
        ];
        let commands = vec![command(Op::RmsNorm { eps: 1e-5 }, &[0, 1], &[2], &[])];
        let ir = command_buffer(
            values,
            commands,
            MemoryPlan::default(),
            &[0],
            &[2],
            Vec::new(),
        );
        validate(&ir);

        let x = vec![0.5, -1.0, 2.0, -0.25];
        let weight = vec![1.0, 0.5, 1.5, 0.75];
        let mut expected = vec![0.0_f32; 4];
        kernels::rms_norm(&x, &weight, &mut expected, 1, 4, 1e-5)
            .expect("direct rms_norm should succeed");

        let bindings = bindings(&[(0, 1), (1, 2), (2, 3)]);
        let mut buffers = BufferStore::new();
        buffers.insert(BufferId(1), x);
        buffers.insert(BufferId(2), weight);
        buffers.insert(BufferId(3), vec![0.0; 4]);
        execute(&ir, &bindings, &mut buffers, &ScalarStore::new())
            .expect("rank-one rms_norm should execute");
        assert_eq!(buffers.get(BufferId(3)).unwrap(), expected.as_slice());
    }
}
