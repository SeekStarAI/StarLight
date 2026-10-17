// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 SeekStarAI and StarLight Contributors

//! [`CpuDevice`]: a [`sl_device::Device`] over the scalar reference kernels.
//!
//! The device stores buffers as `Vec<Option<Vec<f32>>>` / `Vec<Option<Vec<u32>>>`
//! and dispatches one operator at a time to [`crate::reference`]. Every
//! `dispatch` validates the ports, takes the buffers it needs out of the
//! store, runs the kernel, and puts them back; no heap allocation happens on
//! that path.
//!
//! Buffers that are addressed by several ports of one command (the `sl-core`
//! scratchpad is the common case) are split with `split_at_mut` after checking
//! that the element ranges do not overlap, so the implementation needs neither
//! `unsafe` nor an intermediate copy.
//!
//! `allocate_*` accepts any power-of-two `align` but only guarantees the
//! natural alignment of `Vec<f32>`/`Vec<u32>`; the scalar kernels have no
//! alignment requirement beyond that.

use sl_device::{BufferRef, Device, DeviceError};
use sl_ir::{BufferId, DType, Op, RopeVariant, Value, ValueRole};

use crate::reference::{self, KernelError};

const MAX_INPUTS: usize = 4;
const MAX_OUTPUTS: usize = 1;
const MAX_PORTS: usize = MAX_INPUTS + MAX_OUTPUTS;
const MAX_RANK: usize = 8;

/// CPU implementation of the StarLight device contract.
///
/// All storage is ordinary host memory. The device is intentionally simple:
/// one operator per [`Device::dispatch`], synchronous completion.
#[derive(Debug, Default)]
pub struct CpuDevice {
    floats: Vec<Option<Vec<f32>>>,
    integers: Vec<Option<Vec<u32>>>,
}

impl CpuDevice {
    /// Creates an empty device.
    pub fn new() -> Self {
        Self::default()
    }

    fn next_buffer_id(&self) -> Result<BufferId, DeviceError> {
        u32::try_from(self.floats.len())
            .map(BufferId)
            .map_err(|_| DeviceError::OutOfMemory {
                requested: self.floats.len(),
            })
    }

    fn allocate(
        &mut self,
        elements: usize,
        align: usize,
        floats: bool,
    ) -> Result<BufferId, DeviceError> {
        validate_alignment(align)?;
        let buffer = self.next_buffer_id()?;
        if floats {
            let mut data = Vec::new();
            data.try_reserve_exact(elements)
                .map_err(|_| DeviceError::OutOfMemory {
                    requested: elements,
                })?;
            data.resize(elements, 0.0);
            self.floats.push(Some(data));
            self.integers.push(None);
        } else {
            let mut data = Vec::new();
            data.try_reserve_exact(elements)
                .map_err(|_| DeviceError::OutOfMemory {
                    requested: elements,
                })?;
            data.resize(elements, 0_u32);
            self.floats.push(None);
            self.integers.push(Some(data));
        }
        Ok(buffer)
    }

    fn f32_slot(&self, buffer: BufferId) -> Result<&[f32], DeviceError> {
        self.floats
            .get(buffer.0 as usize)
            .and_then(Option::as_deref)
            .ok_or(DeviceError::InvalidBuffer(buffer))
    }

    fn f32_slot_mut(&mut self, buffer: BufferId) -> Result<&mut Vec<f32>, DeviceError> {
        self.floats
            .get_mut(buffer.0 as usize)
            .and_then(Option::as_mut)
            .ok_or(DeviceError::InvalidBuffer(buffer))
    }

    fn u32_slot(&self, buffer: BufferId) -> Result<&[u32], DeviceError> {
        self.integers
            .get(buffer.0 as usize)
            .and_then(Option::as_deref)
            .ok_or(DeviceError::InvalidBuffer(buffer))
    }

    fn u32_slot_mut(&mut self, buffer: BufferId) -> Result<&mut Vec<u32>, DeviceError> {
        self.integers
            .get_mut(buffer.0 as usize)
            .and_then(Option::as_mut)
            .ok_or(DeviceError::InvalidBuffer(buffer))
    }
}

impl Device for CpuDevice {
    fn allocate_f32(&mut self, elements: usize, align: usize) -> Result<BufferId, DeviceError> {
        self.allocate(elements, align, true)
    }

    fn allocate_u32(&mut self, elements: usize, align: usize) -> Result<BufferId, DeviceError> {
        self.allocate(elements, align, false)
    }

    fn upload_f32(&mut self, buffer: BufferId, data: &[f32]) -> Result<(), DeviceError> {
        let slot = self.f32_slot_mut(buffer)?;
        if slot.len() != data.len() {
            return Err(DeviceError::InvalidRange {
                buffer,
                offset: 0,
                len: data.len(),
            });
        }
        slot.copy_from_slice(data);
        Ok(())
    }

    fn upload_u32(&mut self, buffer: BufferId, data: &[u32]) -> Result<(), DeviceError> {
        let slot = self.u32_slot_mut(buffer)?;
        if slot.len() != data.len() {
            return Err(DeviceError::InvalidRange {
                buffer,
                offset: 0,
                len: data.len(),
            });
        }
        slot.copy_from_slice(data);
        Ok(())
    }

    fn download_f32(&mut self, buffer: BufferId, out: &mut [f32]) -> Result<(), DeviceError> {
        let slot = self.f32_slot(buffer)?;
        if slot.len() != out.len() {
            return Err(DeviceError::InvalidRange {
                buffer,
                offset: 0,
                len: out.len(),
            });
        }
        out.copy_from_slice(slot);
        Ok(())
    }

    fn download_u32(&mut self, buffer: BufferId, out: &mut [u32]) -> Result<(), DeviceError> {
        let slot = self.u32_slot(buffer)?;
        if slot.len() != out.len() {
            return Err(DeviceError::InvalidRange {
                buffer,
                offset: 0,
                len: out.len(),
            });
        }
        out.copy_from_slice(slot);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch(
        &mut self,
        op: &Op,
        inputs: &[BufferRef],
        input_values: &[&Value],
        outputs: &[BufferRef],
        output_values: &[&Value],
        scalars: &[u32],
    ) -> Result<(), DeviceError> {
        validate_call(op, inputs, input_values, outputs, output_values, scalars)?;
        for (reference, value) in inputs.iter().zip(input_values) {
            validate_port(self, *reference, value, false)?;
        }
        for (reference, value) in outputs.iter().zip(output_values) {
            validate_port(self, *reference, value, true)?;
        }
        check_disjoint(inputs, outputs)?;

        let output_buffer = outputs.first().map(|reference| reference.buffer);
        let state_index = input_values
            .iter()
            .position(|value| value.role == ValueRole::State);

        let mut stash = Stash::new(state_index, output_buffer);
        if let Err(error) = stash.take(self, inputs, input_values) {
            stash.restore(self);
            return Err(error);
        }

        let result = stash.run(op, inputs, outputs, input_values, output_values, scalars);
        stash.restore(self);
        result
    }

    fn synchronize(&mut self) -> Result<(), DeviceError> {
        Ok(())
    }
}

fn validate_alignment(align: usize) -> Result<(), DeviceError> {
    if align == 0 || !align.is_power_of_two() {
        return Err(DeviceError::Execution(
            "alignment must be a nonzero power of two",
        ));
    }
    Ok(())
}

fn validate_call(
    op: &Op,
    inputs: &[BufferRef],
    input_values: &[&Value],
    outputs: &[BufferRef],
    output_values: &[&Value],
    scalars: &[u32],
) -> Result<(), DeviceError> {
    if inputs.len() != input_values.len() {
        return Err(DeviceError::Execution(
            "inputs and input_values have different lengths",
        ));
    }
    if outputs.len() != output_values.len() {
        return Err(DeviceError::Execution(
            "outputs and output_values have different lengths",
        ));
    }
    if inputs.len() > MAX_INPUTS || outputs.len() > MAX_OUTPUTS {
        return Err(DeviceError::Execution("too many command ports"));
    }

    let signature = op.code().signature();
    let scalar_ports = signature
        .inputs
        .iter()
        .filter(|port| port.roles.contains(ValueRole::Scalar))
        .count();
    let state_ports = input_values
        .iter()
        .filter(|value| value.role == ValueRole::State)
        .count();
    let stateful = matches!(op, Op::KvAppend { .. } | Op::CausalAttention { .. });
    if stateful {
        if state_ports != 1 {
            return Err(DeviceError::Execution(
                "stateful operator requires exactly one State input",
            ));
        }
    } else if state_ports != 0 {
        return Err(DeviceError::Execution(
            "stateless operator received a State input",
        ));
    }

    let expected_inputs = signature.inputs.len() - scalar_ports + state_ports;
    if inputs.len() != expected_inputs {
        return Err(DeviceError::Execution(
            "input buffer count does not match the operator signature",
        ));
    }
    if outputs.len() != signature.outputs.len() {
        return Err(DeviceError::Execution(
            "output count does not match the operator signature",
        ));
    }
    if scalars.len() != scalar_ports {
        return Err(DeviceError::Execution(
            "scalar count does not match the operator signature",
        ));
    }
    Ok(())
}

fn validate_port(
    device: &CpuDevice,
    reference: BufferRef,
    value: &Value,
    output: bool,
) -> Result<(), DeviceError> {
    if value.role == ValueRole::Scalar {
        return Err(DeviceError::Execution(
            "scalar operands must be passed through the scalars argument",
        ));
    }
    let dims = Dims::from_value(value)?;
    let elements = dims.element_count()?;
    if elements != reference.len {
        return Err(DeviceError::Execution(
            "BufferRef length does not match the value element count",
        ));
    }
    match value.dtype {
        DType::F32 => {
            let capacity = device.f32_slot(reference.buffer)?.len();
            check_range(reference, capacity)
        }
        DType::U32 => {
            if output {
                return Err(DeviceError::Execution(
                    "integer command outputs are not supported",
                ));
            }
            let capacity = device.u32_slot(reference.buffer)?.len();
            check_range(reference, capacity)
        }
        _ => Err(DeviceError::Unimplemented(
            "dtype is not supported by CpuDevice",
        )),
    }
}

fn check_range(reference: BufferRef, capacity: usize) -> Result<(), DeviceError> {
    let end = reference
        .offset
        .checked_add(reference.len)
        .ok_or(DeviceError::ShapeOverflow)?;
    if end > capacity {
        return Err(DeviceError::InvalidRange {
            buffer: reference.buffer,
            offset: reference.offset,
            len: reference.len,
        });
    }
    Ok(())
}

fn check_disjoint(inputs: &[BufferRef], outputs: &[BufferRef]) -> Result<(), DeviceError> {
    for (index, left) in inputs.iter().enumerate() {
        for right in &inputs[index + 1..] {
            if ranges_overlap(*left, *right) {
                return Err(DeviceError::Execution(
                    "one command aliases overlapping input ranges",
                ));
            }
        }
        for right in outputs {
            if ranges_overlap(*left, *right) {
                return Err(DeviceError::Execution(
                    "a command output aliases an overlapping input range",
                ));
            }
        }
    }
    for (index, left) in outputs.iter().enumerate() {
        for right in &outputs[index + 1..] {
            if ranges_overlap(*left, *right) {
                return Err(DeviceError::Execution(
                    "one command aliases overlapping output ranges",
                ));
            }
        }
    }
    Ok(())
}

fn ranges_overlap(left: BufferRef, right: BufferRef) -> bool {
    if left.buffer != right.buffer {
        return false;
    }
    let left_end = left.offset.saturating_add(left.len);
    let right_end = right.offset.saturating_add(right.len);
    left.offset < right_end && right.offset < left_end
}

/// Buffers taken out of the device store for the duration of one command.
struct Stash {
    state: Option<(BufferId, Vec<f32>)>,
    output: Option<(BufferId, Vec<f32>)>,
    f32_inputs: [Option<(BufferId, Vec<f32>)>; MAX_INPUTS],
    u32_inputs: [Option<(BufferId, Vec<u32>)>; MAX_INPUTS],
    f32_slots: [usize; MAX_INPUTS],
    u32_slots: [usize; MAX_INPUTS],
    f32_count: usize,
    u32_count: usize,
    state_index: Option<usize>,
    output_buffer: Option<BufferId>,
}

impl Stash {
    fn new(state_index: Option<usize>, output_buffer: Option<BufferId>) -> Self {
        Self {
            state: None,
            output: None,
            f32_inputs: core::array::from_fn(|_| None),
            u32_inputs: core::array::from_fn(|_| None),
            f32_slots: [usize::MAX; MAX_INPUTS],
            u32_slots: [usize::MAX; MAX_INPUTS],
            f32_count: 0,
            u32_count: 0,
            state_index,
            output_buffer,
        }
    }

    fn take(
        &mut self,
        device: &mut CpuDevice,
        inputs: &[BufferRef],
        input_values: &[&Value],
    ) -> Result<(), DeviceError> {
        if let Some(index) = self.state_index {
            let buffer = inputs[index].buffer;
            let data = device
                .floats
                .get_mut(buffer.0 as usize)
                .and_then(Option::take)
                .ok_or(DeviceError::InvalidBuffer(buffer))?;
            self.state = Some((buffer, data));
        }
        if let Some(buffer) = self.output_buffer {
            let data = device
                .floats
                .get_mut(buffer.0 as usize)
                .and_then(Option::take)
                .ok_or(DeviceError::InvalidBuffer(buffer))?;
            self.output = Some((buffer, data));
        }

        for (index, (reference, value)) in inputs.iter().zip(input_values).enumerate() {
            if self.state_index == Some(index) {
                continue;
            }
            if self.output_buffer == Some(reference.buffer) {
                continue;
            }
            match value.dtype {
                DType::F32 => {
                    let slot = match self.f32_inputs[..self.f32_count].iter().position(|entry| {
                        entry
                            .as_ref()
                            .is_some_and(|(id, _)| *id == reference.buffer)
                    }) {
                        Some(slot) => slot,
                        None => {
                            let data = device
                                .floats
                                .get_mut(reference.buffer.0 as usize)
                                .and_then(Option::take)
                                .ok_or(DeviceError::InvalidBuffer(reference.buffer))?;
                            let slot = self.f32_count;
                            if slot >= MAX_INPUTS {
                                return Err(DeviceError::Execution("too many f32 inputs"));
                            }
                            self.f32_inputs[slot] = Some((reference.buffer, data));
                            self.f32_count += 1;
                            slot
                        }
                    };
                    self.f32_slots[index] = slot;
                }
                DType::U32 => {
                    let slot = match self.u32_inputs[..self.u32_count].iter().position(|entry| {
                        entry
                            .as_ref()
                            .is_some_and(|(id, _)| *id == reference.buffer)
                    }) {
                        Some(slot) => slot,
                        None => {
                            let data = device
                                .integers
                                .get_mut(reference.buffer.0 as usize)
                                .and_then(Option::take)
                                .ok_or(DeviceError::InvalidBuffer(reference.buffer))?;
                            let slot = self.u32_count;
                            if slot >= MAX_INPUTS {
                                return Err(DeviceError::Execution("too many u32 inputs"));
                            }
                            self.u32_inputs[slot] = Some((reference.buffer, data));
                            self.u32_count += 1;
                            slot
                        }
                    };
                    self.u32_slots[index] = slot;
                }
                _ => {
                    return Err(DeviceError::Unimplemented(
                        "dtype is not supported by CpuDevice",
                    ))
                }
            }
        }
        Ok(())
    }

    fn restore(self, device: &mut CpuDevice) {
        if let Some((buffer, data)) = self.state {
            if let Some(slot) = device.floats.get_mut(buffer.0 as usize) {
                *slot = Some(data);
            }
        }
        if let Some((buffer, data)) = self.output {
            if let Some(slot) = device.floats.get_mut(buffer.0 as usize) {
                *slot = Some(data);
            }
        }
        for (buffer, data) in self.f32_inputs.into_iter().flatten() {
            if let Some(slot) = device.floats.get_mut(buffer.0 as usize) {
                *slot = Some(data);
            }
        }
        for (buffer, data) in self.u32_inputs.into_iter().flatten() {
            if let Some(slot) = device.integers.get_mut(buffer.0 as usize) {
                *slot = Some(data);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        op: &Op,
        inputs: &[BufferRef],
        outputs: &[BufferRef],
        input_values: &[&Value],
        output_values: &[&Value],
        scalars: &[u32],
    ) -> Result<(), DeviceError> {
        let mut f32_inputs: [Option<&[f32]>; MAX_INPUTS] = [None; MAX_INPUTS];
        let mut u32_inputs: [Option<&[u32]>; MAX_INPUTS] = [None; MAX_INPUTS];

        for (index, reference) in inputs.iter().enumerate() {
            if self.state_index == Some(index) {
                continue;
            }
            if self.output_buffer == Some(reference.buffer) {
                continue;
            }
            let value = input_values[index];
            match value.dtype {
                DType::F32 => {
                    let slot = self.f32_slots[index];
                    let data = self.f32_inputs[slot]
                        .as_ref()
                        .map(|(_, data)| data.as_slice())
                        .ok_or(DeviceError::Execution("f32 input was not staged"))?;
                    f32_inputs[index] = Some(
                        data.get(reference.offset..reference.offset + reference.len)
                            .ok_or(DeviceError::InvalidRange {
                                buffer: reference.buffer,
                                offset: reference.offset,
                                len: reference.len,
                            })?,
                    );
                }
                DType::U32 => {
                    let slot = self.u32_slots[index];
                    let data = self.u32_inputs[slot]
                        .as_ref()
                        .map(|(_, data)| data.as_slice())
                        .ok_or(DeviceError::Execution("u32 input was not staged"))?;
                    u32_inputs[index] = Some(
                        data.get(reference.offset..reference.offset + reference.len)
                            .ok_or(DeviceError::InvalidRange {
                                buffer: reference.buffer,
                                offset: reference.offset,
                                len: reference.len,
                            })?,
                    );
                }
                _ => {
                    return Err(DeviceError::Unimplemented(
                        "dtype is not supported by CpuDevice",
                    ))
                }
            }
        }

        let mut shared_ranges = [(0_usize, 0_usize, 0_usize); MAX_PORTS];
        let mut shared_count = 0_usize;
        for (index, reference) in inputs.iter().enumerate() {
            if self.state_index == Some(index) {
                continue;
            }
            if self.output_buffer == Some(reference.buffer) {
                shared_ranges[shared_count] = (reference.offset, reference.len, index);
                shared_count += 1;
            }
        }
        if let Some(buffer) = self.output_buffer {
            if let Some(output) = outputs.first() {
                shared_ranges[shared_count] = (output.offset, output.len, inputs.len());
                shared_count += 1;
            }
            let _ = buffer;
        }

        let mut empty_output: [f32; 0] = [];
        let output_slice: &mut [f32] = if shared_count == 0 {
            match outputs.first() {
                Some(reference) => {
                    let (_, data) = self
                        .output
                        .as_mut()
                        .ok_or(DeviceError::Execution("output buffer was not staged"))?;
                    data.get_mut(reference.offset..reference.offset + reference.len)
                        .ok_or(DeviceError::InvalidRange {
                            buffer: reference.buffer,
                            offset: reference.offset,
                            len: reference.len,
                        })?
                }
                None => &mut empty_output,
            }
        } else {
            let buffer = self
                .output_buffer
                .ok_or(DeviceError::Execution("shared output buffer is missing"))?;
            let data = self
                .output
                .as_mut()
                .map(|(_, data)| data.as_mut_slice())
                .ok_or(DeviceError::Execution("output buffer was not staged"))?;
            let mut pieces = split_ranges(data, buffer, &mut shared_ranges, shared_count)?;
            for (_, _, port) in &shared_ranges[..shared_count] {
                if *port < inputs.len() {
                    if let Some(piece) = pieces[*port].take() {
                        f32_inputs[*port] = Some(piece);
                    }
                }
            }
            pieces[inputs.len()]
                .take()
                .ok_or(DeviceError::Execution("output range was not staged"))?
        };

        let state_slice: Option<&mut [f32]> = match self.state_index {
            Some(index) => {
                let reference = inputs[index];
                let (_, data) = self
                    .state
                    .as_mut()
                    .ok_or(DeviceError::Execution("state buffer was not staged"))?;
                Some(
                    data.get_mut(reference.offset..reference.offset + reference.len)
                        .ok_or(DeviceError::InvalidRange {
                            buffer: reference.buffer,
                            offset: reference.offset,
                            len: reference.len,
                        })?,
                )
            }
            None => None,
        };
        let state_value = self
            .state_index
            .and_then(|index| input_values.get(index).copied());

        run_op(
            op,
            input_values,
            output_values,
            &f32_inputs,
            &u32_inputs,
            output_slice,
            state_value,
            state_slice,
            scalars,
        )
    }
}

fn split_ranges<'a>(
    data: &'a mut [f32],
    buffer: BufferId,
    ranges: &mut [(usize, usize, usize); MAX_PORTS],
    count: usize,
) -> Result<[Option<&'a mut [f32]>; MAX_PORTS], DeviceError> {
    ranges[..count].sort_unstable_by_key(|(start, _, _)| *start);
    let data_len = data.len();
    let mut pieces: [Option<&'a mut [f32]>; MAX_PORTS] = core::array::from_fn(|_| None);
    let mut rest: &'a mut [f32] = data;
    let mut cursor = 0_usize;
    for &(start, len, port) in &ranges[..count] {
        if start < cursor {
            return Err(DeviceError::Execution("scratchpad port ranges overlap"));
        }
        let end = start.checked_add(len).ok_or(DeviceError::ShapeOverflow)?;
        if end > data_len {
            return Err(DeviceError::InvalidRange {
                buffer,
                offset: start,
                len,
            });
        }
        let (_, tail) = rest.split_at_mut(start - cursor);
        let (piece, tail) = tail.split_at_mut(len);
        pieces[port] = Some(piece);
        rest = tail;
        cursor = end;
    }
    Ok(pieces)
}

/// A resolved shape: rank plus concrete extents.
#[derive(Clone, Copy, Debug, Default)]
struct Dims {
    rank: usize,
    extents: [usize; MAX_RANK],
}

impl Dims {
    fn from_value(value: &Value) -> Result<Self, DeviceError> {
        let shape = value.shape.as_slice();
        if shape.len() > MAX_RANK {
            return Err(DeviceError::ShapeOverflow);
        }
        let mut extents = [0_usize; MAX_RANK];
        for (index, dimension) in shape.iter().enumerate() {
            let extent = dimension.as_const().ok_or(DeviceError::Execution(
                "value shape still contains a symbolic dimension",
            ))?;
            extents[index] = usize::try_from(extent).map_err(|_| DeviceError::ShapeOverflow)?;
        }
        Ok(Self {
            rank: shape.len(),
            extents,
        })
    }

    fn dim(&self, index: usize) -> Result<usize, DeviceError> {
        if index < self.rank {
            Ok(self.extents[index])
        } else {
            Err(invalid_shape("operator rank is too small"))
        }
    }

    fn last(&self) -> Result<usize, DeviceError> {
        if self.rank == 0 {
            Err(invalid_shape("operator requires at least one dimension"))
        } else {
            Ok(self.extents[self.rank - 1])
        }
    }

    fn product(&self, end: usize) -> Result<usize, DeviceError> {
        if end > self.rank {
            return Err(invalid_shape("operator rank is too small"));
        }
        self.extents[..end]
            .iter()
            .try_fold(1_usize, |product, &extent| {
                product
                    .checked_mul(extent)
                    .ok_or(DeviceError::ShapeOverflow)
            })
    }

    fn element_count(&self) -> Result<usize, DeviceError> {
        self.product(self.rank)
    }

    fn same_as(&self, other: &Self) -> bool {
        self.rank == other.rank && self.extents[..self.rank] == other.extents[..other.rank]
    }
}

#[allow(clippy::too_many_arguments)]
fn run_op(
    op: &Op,
    input_values: &[&Value],
    output_values: &[&Value],
    f32_inputs: &[Option<&[f32]>; MAX_INPUTS],
    u32_inputs: &[Option<&[u32]>; MAX_INPUTS],
    output: &mut [f32],
    state_value: Option<&Value>,
    state: Option<&mut [f32]>,
    scalars: &[u32],
) -> Result<(), DeviceError> {
    match op {
        Op::MatMulNT | Op::MatMulNN => {
            let transposed = matches!(op, Op::MatMulNT);
            let left_dims = dims(ivalue(input_values, 0)?)?;
            let right_dims = dims(ivalue(input_values, 1)?)?;
            let out_dims = dims(ovalue(output_values, 0)?)?;
            let (batch_a, batch_b, m, k, n) =
                matmul_geometry(&left_dims, &right_dims, &out_dims, transposed)?;
            matmul_broadcast(
                f32_input(f32_inputs, 0)?,
                f32_input(f32_inputs, 1)?,
                output,
                batch_a,
                batch_b,
                m,
                k,
                n,
                transposed,
            )
        }
        Op::Softmax => {
            let input_dims = dims(ivalue(input_values, 0)?)?;
            let out_dims = dims(ovalue(output_values, 0)?)?;
            if !input_dims.same_as(&out_dims) {
                return Err(invalid_shape("softmax output shape must match its input"));
            }
            let cols = input_dims.last()?;
            let rows = input_dims.product(input_dims.rank - 1)?;
            let values = f32_input(f32_inputs, 0)?;
            copy_into(values, output)?;
            reference::softmax(output, rows, cols).map_err(kernel_error)
        }
        Op::SoftmaxCausal => {
            let input_dims = dims(ivalue(input_values, 0)?)?;
            let out_dims = dims(ovalue(output_values, 0)?)?;
            if !input_dims.same_as(&out_dims) {
                return Err(invalid_shape(
                    "softmax_causal output shape must match its input",
                ));
            }
            let cols = input_dims.last()?;
            let rows = input_dims.product(input_dims.rank - 1)?;
            let seq = scalar_at(scalars, 0)?;
            let past_len = scalar_at(scalars, 1)?;
            let values = f32_input(f32_inputs, 0)?;
            copy_into(values, output)?;
            reference::softmax_causal(output, rows, cols, seq, past_len).map_err(kernel_error)
        }
        Op::RmsNorm { eps } => {
            let input_dims = dims(ivalue(input_values, 0)?)?;
            let weight_dims = dims(ivalue(input_values, 1)?)?;
            let out_dims = dims(ovalue(output_values, 0)?)?;
            let (rows, hidden) = row_geometry(&input_dims)?;
            if weight_dims.rank != 1 || weight_dims.dim(0)? != hidden {
                return Err(invalid_shape("rms_norm weight shape must be [hidden]"));
            }
            if !input_dims.same_as(&out_dims) {
                return Err(invalid_shape("rms_norm output shape must match its input"));
            }
            reference::rms_norm(
                f32_input(f32_inputs, 0)?,
                f32_input(f32_inputs, 1)?,
                output,
                rows,
                hidden,
                *eps,
            )
            .map_err(kernel_error)
        }
        Op::Rope { variant, theta } => {
            let input_dims = dims(ivalue(input_values, 0)?)?;
            let position_dims = dims(ivalue(input_values, 1)?)?;
            let out_dims = dims(ovalue(output_values, 0)?)?;
            let (batch, heads, seq, head_dim) = rope_geometry(&input_dims, &position_dims)?;
            if !input_dims.same_as(&out_dims) {
                return Err(invalid_shape("rope output shape must match its input"));
            }
            let values = f32_input(f32_inputs, 0)?;
            let positions = u32_input(u32_inputs, 1)?;
            copy_into(values, output)?;
            match variant {
                RopeVariant::Neox => {
                    reference::rope_neox(output, positions, batch, heads, seq, head_dim, *theta)
                }
                RopeVariant::GptJ => {
                    reference::rope_gptj(output, positions, batch, heads, seq, head_dim, *theta)
                }
            }
            .map_err(kernel_error)
        }
        Op::Silu => {
            let input_dims = dims(ivalue(input_values, 0)?)?;
            let out_dims = dims(ovalue(output_values, 0)?)?;
            if !input_dims.same_as(&out_dims) {
                return Err(invalid_shape("silu output shape must match its input"));
            }
            copy_into(f32_input(f32_inputs, 0)?, output)?;
            reference::silu(output).map_err(kernel_error)
        }
        Op::Mul | Op::Add => {
            let left_dims = dims(ivalue(input_values, 0)?)?;
            let right_dims = dims(ivalue(input_values, 1)?)?;
            let out_dims = dims(ovalue(output_values, 0)?)?;
            if !left_dims.same_as(&right_dims) || !left_dims.same_as(&out_dims) {
                return Err(invalid_shape(
                    "mul/add operands and output must have identical shapes",
                ));
            }
            let len = left_dims.element_count()?;
            let left = f32_input(f32_inputs, 0)?;
            let right = f32_input(f32_inputs, 1)?;
            if matches!(op, Op::Mul) {
                reference::mul(left, right, output, len).map_err(kernel_error)
            } else {
                reference::add(left, right, output, len).map_err(kernel_error)
            }
        }
        Op::Scale { factor } => {
            let input_dims = dims(ivalue(input_values, 0)?)?;
            let out_dims = dims(ovalue(output_values, 0)?)?;
            if !input_dims.same_as(&out_dims) {
                return Err(invalid_shape("scale output shape must match its input"));
            }
            copy_into(f32_input(f32_inputs, 0)?, output)?;
            reference::scale(output, *factor).map_err(kernel_error)
        }
        Op::Gather => {
            let table_dims = dims(ivalue(input_values, 0)?)?;
            let id_dims = dims(ivalue(input_values, 1)?)?;
            let out_dims = dims(ovalue(output_values, 0)?)?;
            if table_dims.rank != 2 {
                return Err(invalid_shape(
                    "gather table must have shape [vocab, hidden]",
                ));
            }
            let vocab = table_dims.dim(0)?;
            let hidden = table_dims.dim(1)?;
            let rows = id_dims.element_count()?;
            if out_dims.rank != id_dims.rank + 1 {
                return Err(invalid_shape(
                    "gather output rank must be the id rank plus one",
                ));
            }
            for index in 0..id_dims.rank {
                if out_dims.dim(index)? != id_dims.dim(index)? {
                    return Err(invalid_shape(
                        "gather output prefix must match the id shape",
                    ));
                }
            }
            if out_dims.last()? != hidden {
                return Err(invalid_shape("gather output width must equal hidden"));
            }
            reference::gather(
                f32_input(f32_inputs, 0)?,
                u32_input(u32_inputs, 1)?,
                output,
                vocab,
                hidden,
                rows,
            )
            .map_err(kernel_error)
        }
        Op::LastToken => {
            let input_dims = dims(ivalue(input_values, 0)?)?;
            let out_dims = dims(ovalue(output_values, 0)?)?;
            let seq_len = scalar_at(scalars, 0)?;
            if input_dims.rank != 3 || out_dims.rank != 2 {
                return Err(invalid_shape(
                    "last_token shapes must be [B, M, H] -> [B, H]",
                ));
            }
            let batch = input_dims.dim(0)?;
            let seq = input_dims.dim(1)?;
            let hidden = input_dims.dim(2)?;
            if out_dims.dim(0)? != batch || out_dims.dim(1)? != hidden {
                return Err(invalid_shape(
                    "last_token output shape must be [batch, hidden]",
                ));
            }
            reference::last_token(
                f32_input(f32_inputs, 0)?,
                output,
                batch,
                seq,
                hidden,
                seq_len,
            )
            .map_err(kernel_error)
        }
        Op::KvAppend { layer } => {
            let key_dims = dims(ivalue(input_values, 0)?)?;
            let value_dims = dims(ivalue(input_values, 1)?)?;
            let past_len = scalar_at(scalars, 0)?;
            let layer = usize::try_from(*layer).map_err(|_| DeviceError::ShapeOverflow)?;
            let (batch, kv_heads, seq, head_dim) = attention_kv_geometry(&key_dims, &value_dims)?;
            let state_value =
                state_value.ok_or(DeviceError::Execution("kv_append requires a State input"))?;
            let state_dims = dims(state_value)?;
            let (layers, max_seq) =
                check_state_geometry(&state_dims, batch, kv_heads, head_dim, layer)?;
            let cache = state.ok_or(DeviceError::Execution(
                "kv_append state buffer was not staged",
            ))?;
            reference::kv_append(
                f32_input(f32_inputs, 0)?,
                f32_input(f32_inputs, 1)?,
                cache,
                batch,
                layers,
                layer,
                kv_heads,
                head_dim,
                max_seq,
                past_len,
                seq,
            )
            .map_err(kernel_error)
        }
        Op::SplitHeads { heads, head_dim } => {
            let input_dims = dims(ivalue(input_values, 0)?)?;
            let out_dims = dims(ovalue(output_values, 0)?)?;
            let heads = usize::try_from(*heads).map_err(|_| DeviceError::ShapeOverflow)?;
            let head_dim = usize::try_from(*head_dim).map_err(|_| DeviceError::ShapeOverflow)?;
            let (batch, seq) = split_geometry(&input_dims, heads, head_dim)?;
            if out_dims.rank != 4
                || out_dims.dim(0)? != batch
                || out_dims.dim(1)? != heads
                || out_dims.dim(2)? != seq
                || out_dims.dim(3)? != head_dim
            {
                return Err(invalid_shape(
                    "split_heads output shape must be [batch, heads, seq, head_dim]",
                ));
            }
            reference::split_heads(
                f32_input(f32_inputs, 0)?,
                output,
                batch,
                seq,
                heads,
                head_dim,
            )
            .map_err(kernel_error)
        }
        Op::MergeHeads { heads, head_dim } => {
            let input_dims = dims(ivalue(input_values, 0)?)?;
            let out_dims = dims(ovalue(output_values, 0)?)?;
            let heads = usize::try_from(*heads).map_err(|_| DeviceError::ShapeOverflow)?;
            let head_dim = usize::try_from(*head_dim).map_err(|_| DeviceError::ShapeOverflow)?;
            if input_dims.rank != 4 {
                return Err(invalid_shape(
                    "merge_heads input shape must be [batch, heads, seq, head_dim]",
                ));
            }
            let batch = input_dims.dim(0)?;
            let seq = input_dims.dim(2)?;
            if input_dims.dim(1)? != heads || input_dims.dim(3)? != head_dim {
                return Err(invalid_shape(
                    "merge_heads input does not match its heads/head_dim attributes",
                ));
            }
            let width = heads
                .checked_mul(head_dim)
                .ok_or(DeviceError::ShapeOverflow)?;
            if out_dims.rank != 3
                || out_dims.dim(0)? != batch
                || out_dims.dim(1)? != seq
                || out_dims.dim(2)? != width
            {
                return Err(invalid_shape(
                    "merge_heads output shape must be [batch, seq, heads * head_dim]",
                ));
            }
            reference::merge_heads(
                f32_input(f32_inputs, 0)?,
                output,
                batch,
                seq,
                heads,
                head_dim,
            )
            .map_err(kernel_error)
        }
        Op::CausalAttention {
            layer,
            heads,
            kv_heads,
            head_dim,
            scale,
        } => {
            let query_dims = dims(ivalue(input_values, 0)?)?;
            let key_dims = dims(ivalue(input_values, 1)?)?;
            let value_dims = dims(ivalue(input_values, 2)?)?;
            let out_dims = dims(ovalue(output_values, 0)?)?;
            let past_len = scalar_at(scalars, 0)?;
            let layer = usize::try_from(*layer).map_err(|_| DeviceError::ShapeOverflow)?;
            let heads = usize::try_from(*heads).map_err(|_| DeviceError::ShapeOverflow)?;
            let kv_heads = usize::try_from(*kv_heads).map_err(|_| DeviceError::ShapeOverflow)?;
            let head_dim = usize::try_from(*head_dim).map_err(|_| DeviceError::ShapeOverflow)?;
            if query_dims.rank != 4
                || key_dims.rank != 4
                || value_dims.rank != 4
                || out_dims.rank != 4
            {
                return Err(invalid_shape("causal_attention operands must have rank 4"));
            }
            let batch = query_dims.dim(0)?;
            let seq = query_dims.dim(2)?;
            if query_dims.dim(1)? != heads || query_dims.dim(3)? != head_dim {
                return Err(invalid_shape(
                    "causal_attention query shape does not match its attributes",
                ));
            }
            if key_dims.dim(0)? != batch
                || value_dims.dim(0)? != batch
                || key_dims.dim(1)? != kv_heads
                || value_dims.dim(1)? != kv_heads
                || key_dims.dim(2)? != seq
                || value_dims.dim(2)? != seq
                || key_dims.dim(3)? != head_dim
                || value_dims.dim(3)? != head_dim
            {
                return Err(invalid_shape(
                    "causal_attention key/value shapes must be [batch, kv_heads, seq, head_dim]",
                ));
            }
            if !out_dims.same_as(&query_dims) {
                return Err(invalid_shape(
                    "causal_attention output shape must match the query shape",
                ));
            }
            let state_value = state_value.ok_or(DeviceError::Execution(
                "causal_attention requires a State input",
            ))?;
            let state_dims = dims(state_value)?;
            let (layers, max_seq) =
                check_state_geometry(&state_dims, batch, kv_heads, head_dim, layer)?;
            let cache = state.ok_or(DeviceError::Execution(
                "causal_attention state buffer was not staged",
            ))?;
            reference::causal_attention(
                f32_input(f32_inputs, 0)?,
                f32_input(f32_inputs, 1)?,
                f32_input(f32_inputs, 2)?,
                cache,
                output,
                batch,
                layers,
                layer,
                heads,
                kv_heads,
                seq,
                past_len,
                head_dim,
                max_seq,
                *scale,
            )
            .map_err(kernel_error)
        }
        Op::SwiGlu => {
            let input_dims = dims(ivalue(input_values, 0)?)?;
            let gate_dims = dims(ivalue(input_values, 1)?)?;
            let up_dims = dims(ivalue(input_values, 2)?)?;
            let down_dims = dims(ivalue(input_values, 3)?)?;
            let out_dims = dims(ovalue(output_values, 0)?)?;
            let (rows, hidden) = row_geometry(&input_dims)?;
            if gate_dims.rank != 2 || up_dims.rank != 2 || down_dims.rank != 2 {
                return Err(invalid_shape("swiglu weights must have rank 2"));
            }
            let inter = gate_dims.dim(0)?;
            if gate_dims.dim(1)? != hidden
                || up_dims.dim(0)? != inter
                || up_dims.dim(1)? != hidden
                || down_dims.dim(0)? != hidden
                || down_dims.dim(1)? != inter
            {
                return Err(invalid_shape(
                    "swiglu weight shapes must be [inter, hidden], [inter, hidden], [hidden, inter]",
                ));
            }
            if !input_dims.same_as(&out_dims) {
                return Err(invalid_shape("swiglu output shape must match its input"));
            }
            reference::swiglu(
                f32_input(f32_inputs, 0)?,
                f32_input(f32_inputs, 1)?,
                f32_input(f32_inputs, 2)?,
                f32_input(f32_inputs, 3)?,
                output,
                rows,
                hidden,
                inter,
            )
            .map_err(kernel_error)
        }
        Op::RmsNormMatMul { eps } => {
            let input_dims = dims(ivalue(input_values, 0)?)?;
            let weight_dims = dims(ivalue(input_values, 1)?)?;
            let matrix_dims = dims(ivalue(input_values, 2)?)?;
            let out_dims = dims(ovalue(output_values, 0)?)?;
            let (rows, hidden) = row_geometry(&input_dims)?;
            if weight_dims.rank != 1 || weight_dims.dim(0)? != hidden {
                return Err(invalid_shape(
                    "rms_norm_matmul weight shape must be [hidden]",
                ));
            }
            if matrix_dims.rank != 2 || matrix_dims.dim(1)? != hidden {
                return Err(invalid_shape(
                    "rms_norm_matmul matrix shape must be [out_features, hidden]",
                ));
            }
            let out_features = matrix_dims.dim(0)?;
            if out_dims.rank != input_dims.rank {
                return Err(invalid_shape(
                    "rms_norm_matmul output rank must match its input rank",
                ));
            }
            for index in 0..input_dims.rank - 1 {
                if out_dims.dim(index)? != input_dims.dim(index)? {
                    return Err(invalid_shape(
                        "rms_norm_matmul output prefix must match its input",
                    ));
                }
            }
            if out_dims.last()? != out_features {
                return Err(invalid_shape(
                    "rms_norm_matmul output width must equal out_features",
                ));
            }
            reference::rms_norm_matmul(
                f32_input(f32_inputs, 0)?,
                f32_input(f32_inputs, 1)?,
                f32_input(f32_inputs, 2)?,
                output,
                rows,
                hidden,
                out_features,
                *eps,
            )
            .map_err(kernel_error)
        }
    }
}

fn dims(value: &Value) -> Result<Dims, DeviceError> {
    Dims::from_value(value)
}

fn ivalue<'a>(values: &'a [&Value], index: usize) -> Result<&'a Value, DeviceError> {
    values
        .get(index)
        .copied()
        .ok_or(DeviceError::Execution("operator is missing an input value"))
}

fn ovalue<'a>(values: &'a [&Value], index: usize) -> Result<&'a Value, DeviceError> {
    values.get(index).copied().ok_or(DeviceError::Execution(
        "operator is missing an output value",
    ))
}

fn f32_input<'a>(
    inputs: &'a [Option<&[f32]>; MAX_INPUTS],
    index: usize,
) -> Result<&'a [f32], DeviceError> {
    inputs
        .get(index)
        .copied()
        .flatten()
        .ok_or(DeviceError::Execution("operator is missing an f32 input"))
}

fn u32_input<'a>(
    inputs: &'a [Option<&[u32]>; MAX_INPUTS],
    index: usize,
) -> Result<&'a [u32], DeviceError> {
    inputs
        .get(index)
        .copied()
        .flatten()
        .ok_or(DeviceError::Execution("operator is missing a u32 input"))
}

fn scalar_at(scalars: &[u32], index: usize) -> Result<usize, DeviceError> {
    let value = scalars
        .get(index)
        .copied()
        .ok_or(DeviceError::Execution("operator is missing a scalar input"))?;
    usize::try_from(value).map_err(|_| DeviceError::ShapeOverflow)
}

fn copy_into(input: &[f32], output: &mut [f32]) -> Result<(), DeviceError> {
    if input.len() != output.len() {
        return Err(DeviceError::Execution(
            "input and output slices have different lengths",
        ));
    }
    output.copy_from_slice(input);
    Ok(())
}

fn kernel_error(error: KernelError) -> DeviceError {
    match error {
        KernelError::LengthMismatch { .. } => {
            DeviceError::Execution("kernel received a slice of the wrong length")
        }
        KernelError::ShapeOverflow => DeviceError::ShapeOverflow,
        KernelError::InvalidShape(reason) => DeviceError::Execution(reason),
        KernelError::Unsupported(reason) => DeviceError::Unimplemented(reason),
    }
}

fn invalid_shape(reason: &'static str) -> DeviceError {
    DeviceError::Execution(reason)
}

fn matmul_geometry(
    left: &Dims,
    right: &Dims,
    out: &Dims,
    transposed: bool,
) -> Result<(usize, usize, usize, usize, usize), DeviceError> {
    if left.rank < 2 || right.rank < 2 || out.rank < 2 {
        return Err(invalid_shape(
            "matmul operands and output must have rank at least 2",
        ));
    }
    let m = left.dim(left.rank - 2)?;
    let k = left.dim(left.rank - 1)?;
    let (right_k, n) = if transposed {
        (right.dim(right.rank - 1)?, right.dim(right.rank - 2)?)
    } else {
        (right.dim(right.rank - 2)?, right.dim(right.rank - 1)?)
    };
    if k != right_k {
        return Err(invalid_shape("matmul reduction dimensions must match"));
    }
    let batch_a = left.product(left.rank - 2)?;
    let batch_b = right.product(right.rank - 2)?;
    let batch = broadcast_batch(batch_a, batch_b)?;
    if out.dim(out.rank - 2)? != m || out.dim(out.rank - 1)? != n {
        return Err(invalid_shape("matmul output shape must be m x n"));
    }
    if out.product(out.rank - 2)? != batch {
        return Err(invalid_shape("matmul output batch must match the operands"));
    }
    Ok((batch_a, batch_b, m, k, n))
}

fn broadcast_batch(left: usize, right: usize) -> Result<usize, DeviceError> {
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
fn matmul_broadcast(
    left: &[f32],
    right: &[f32],
    output: &mut [f32],
    batch_a: usize,
    batch_b: usize,
    m: usize,
    k: usize,
    n: usize,
    transposed: bool,
) -> Result<(), DeviceError> {
    let batch = broadcast_batch(batch_a, batch_b)?;
    if batch == 0 {
        return Ok(());
    }
    if batch_a == batch_b {
        return if transposed {
            reference::matmul_nt(left, right, output, batch, m, k, n).map_err(kernel_error)
        } else {
            reference::matmul_nn(left, right, output, batch, m, k, n).map_err(kernel_error)
        };
    }
    let a_stride = m.checked_mul(k).ok_or(DeviceError::ShapeOverflow)?;
    let b_stride = if transposed {
        n.checked_mul(k).ok_or(DeviceError::ShapeOverflow)?
    } else {
        k.checked_mul(n).ok_or(DeviceError::ShapeOverflow)?
    };
    let out_stride = m.checked_mul(n).ok_or(DeviceError::ShapeOverflow)?;
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
        if transposed {
            reference::matmul_nt(a_slice, b_slice, out_slice, 1, m, k, n).map_err(kernel_error)?;
        } else {
            reference::matmul_nn(a_slice, b_slice, out_slice, 1, m, k, n).map_err(kernel_error)?;
        }
    }
    Ok(())
}

fn row_geometry(input: &Dims) -> Result<(usize, usize), DeviceError> {
    let hidden = input.last()?;
    let rows = input.product(input.rank - 1)?;
    Ok((rows, hidden))
}

fn rope_geometry(
    input: &Dims,
    positions: &Dims,
) -> Result<(usize, usize, usize, usize), DeviceError> {
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
) -> Result<(usize, usize), DeviceError> {
    if input.rank != 3 {
        return Err(invalid_shape(
            "split_heads input shape must be [batch, seq, heads * head_dim]",
        ));
    }
    let batch = input.dim(0)?;
    let seq = input.dim(1)?;
    let width = heads
        .checked_mul(head_dim)
        .ok_or(DeviceError::ShapeOverflow)?;
    if input.dim(2)? != width {
        return Err(invalid_shape(
            "split_heads input width must equal heads * head_dim",
        ));
    }
    Ok((batch, seq))
}

fn attention_kv_geometry(
    key: &Dims,
    value: &Dims,
) -> Result<(usize, usize, usize, usize), DeviceError> {
    if key.rank != 4 || value.rank != 4 {
        return Err(invalid_shape("kv_append key/value must have rank 4"));
    }
    if !key.same_as(value) {
        return Err(invalid_shape("kv_append key and value shapes must match"));
    }
    Ok((key.dim(0)?, key.dim(1)?, key.dim(2)?, key.dim(3)?))
}

fn check_state_geometry(
    state: &Dims,
    batch: usize,
    kv_heads: usize,
    head_dim: usize,
    layer: usize,
) -> Result<(usize, usize), DeviceError> {
    if state.rank != 6
        || state.dim(0)? != batch
        || state.dim(2)? != 2
        || state.dim(4)? != kv_heads
        || state.dim(5)? != head_dim
    {
        return Err(invalid_shape(
            "KV cache shape must be [batch, layers, 2, max_seq, kv_heads, head_dim]",
        ));
    }
    let layers = state.dim(1)?;
    if layer >= layers {
        return Err(invalid_shape("attention layer is out of range"));
    }
    Ok((layers, state.dim(3)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference as kernels;
    use sl_ir::{DType, Shape};

    fn f32_role(shape: &[u32], role: ValueRole) -> Value {
        Value::new(DType::F32, Shape::consts(shape), role)
    }

    fn f32_value(shape: &[u32]) -> Value {
        f32_role(shape, ValueRole::Temp)
    }

    fn u32_value(shape: &[u32]) -> Value {
        Value::new(DType::U32, Shape::consts(shape), ValueRole::Input)
    }

    #[test]
    fn allocate_upload_and_download_round_trip() {
        let mut device = CpuDevice::new();
        let floats = device.allocate_f32(4, 64).expect("allocate f32");
        device
            .upload_f32(floats, &[1.0, -2.0, 3.5, 0.0])
            .expect("upload f32");
        let mut out = [0.0_f32; 4];
        device.download_f32(floats, &mut out).expect("download f32");
        assert_eq!(out, [1.0, -2.0, 3.5, 0.0]);

        let integers = device.allocate_u32(3, 64).expect("allocate u32");
        device.upload_u32(integers, &[7, 8, 9]).expect("upload u32");
        let mut ids = [0_u32; 3];
        device
            .download_u32(integers, &mut ids)
            .expect("download u32");
        assert_eq!(ids, [7, 8, 9]);
    }

    #[test]
    fn rejects_unknown_buffers_and_ranges() {
        let mut device = CpuDevice::new();
        assert_eq!(
            device.upload_f32(BufferId(9), &[1.0]),
            Err(DeviceError::InvalidBuffer(BufferId(9)))
        );
        let floats = device.allocate_f32(2, 64).unwrap();
        assert_eq!(
            device.upload_f32(floats, &[1.0, 2.0, 3.0]),
            Err(DeviceError::InvalidRange {
                buffer: floats,
                offset: 0,
                len: 3,
            })
        );
        assert_eq!(
            device.download_f32(floats, &mut [0.0]),
            Err(DeviceError::InvalidRange {
                buffer: floats,
                offset: 0,
                len: 1,
            })
        );
        assert_eq!(
            device.allocate_f32(1, 3),
            Err(DeviceError::Execution(
                "alignment must be a nonzero power of two"
            ))
        );
    }

    fn logits_after_chain() -> (CpuDevice, BufferId, [f32; 8]) {
        let mut device = CpuDevice::new();
        let table = device.allocate_f32(8 * 4, 64).unwrap();
        let norm = device.allocate_f32(4, 64).unwrap();
        let matmul = device.allocate_f32(8 * 4, 64).unwrap();
        let scratch = device.allocate_f32(48, 64).unwrap();
        let ids = device.allocate_u32(2, 64).unwrap();
        let logits = device.allocate_f32(8, 64).unwrap();

        let table_data: Vec<f32> = (0..32).map(|index| index as f32 * 0.1).collect();
        let norm_data = vec![1.0, 0.5, 1.5, 0.75];
        let matmul_data: Vec<f32> = (0..32)
            .map(|index| ((index % 7) as f32 - 3.0) * 0.2)
            .collect();
        device.upload_f32(table, &table_data).unwrap();
        device.upload_f32(norm, &norm_data).unwrap();
        device.upload_f32(matmul, &matmul_data).unwrap();
        device.upload_u32(ids, &[3, 5]).unwrap();

        let token_value = u32_value(&[1, 2]);
        let table_value = Value::new(DType::F32, Shape::consts(&[8, 4]), ValueRole::Param);
        let gather_value = f32_value(&[1, 2, 4]);
        let norm_value = Value::new(DType::F32, Shape::consts(&[4]), ValueRole::Param);
        let norm_out = f32_value(&[1, 2, 4]);
        let matmul_value = Value::new(DType::F32, Shape::consts(&[8, 4]), ValueRole::Param);
        let matmul_out = f32_value(&[1, 2, 8]);
        let logits_value = Value::new(DType::F32, Shape::consts(&[1, 8]), ValueRole::Output);

        let table_ref = BufferRef {
            buffer: table,
            offset: 0,
            len: 32,
        };
        let norm_ref = BufferRef {
            buffer: norm,
            offset: 0,
            len: 4,
        };
        let matmul_ref = BufferRef {
            buffer: matmul,
            offset: 0,
            len: 32,
        };
        let gather_ref = BufferRef {
            buffer: scratch,
            offset: 0,
            len: 8,
        };
        let norm_out_ref = BufferRef {
            buffer: scratch,
            offset: 16,
            len: 8,
        };
        let matmul_out_ref = BufferRef {
            buffer: scratch,
            offset: 32,
            len: 16,
        };
        let ids_ref = BufferRef {
            buffer: ids,
            offset: 0,
            len: 2,
        };
        let logits_ref = BufferRef {
            buffer: logits,
            offset: 0,
            len: 8,
        };

        device
            .dispatch(
                &Op::Gather,
                &[table_ref, ids_ref],
                &[&table_value, &token_value],
                &[gather_ref],
                &[&gather_value],
                &[],
            )
            .expect("gather dispatch");
        device
            .dispatch(
                &Op::RmsNorm { eps: 1e-5 },
                &[gather_ref, norm_ref],
                &[&gather_value, &norm_value],
                &[norm_out_ref],
                &[&norm_out],
                &[],
            )
            .expect("rms_norm dispatch");
        device
            .dispatch(
                &Op::MatMulNT,
                &[norm_out_ref, matmul_ref],
                &[&norm_out, &matmul_value],
                &[matmul_out_ref],
                &[&matmul_out],
                &[],
            )
            .expect("matmul dispatch");
        device
            .dispatch(
                &Op::LastToken,
                &[matmul_out_ref],
                &[&matmul_out],
                &[logits_ref],
                &[&logits_value],
                &[2],
            )
            .expect("last_token dispatch");

        let mut expected_gather = vec![0.0_f32; 8];
        kernels::gather(&table_data, &[3, 5], &mut expected_gather, 8, 4, 2).unwrap();
        let mut expected_norm = vec![0.0_f32; 8];
        kernels::rms_norm(&expected_gather, &norm_data, &mut expected_norm, 2, 4, 1e-5).unwrap();
        let mut expected_matmul = vec![0.0_f32; 16];
        kernels::matmul_nt(
            &expected_norm,
            &matmul_data,
            &mut expected_matmul,
            1,
            2,
            4,
            8,
        )
        .unwrap();
        let mut expected = [0.0_f32; 8];
        kernels::last_token(&expected_matmul, &mut expected, 1, 2, 8, 2).unwrap();

        (device, logits, expected)
    }

    #[test]
    fn dispatches_gather_rms_norm_matmul_and_last_token() {
        let (mut device, logits, expected) = logits_after_chain();
        let mut out = [0.0_f32; 8];
        // The chain test above already dispatched through the device; read the
        // final logits back to prove the offsets and slicing are correct.
        device.download_f32(logits, &mut out).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn dispatches_elementwise_ops() {
        let mut device = CpuDevice::new();
        let a_data = [0.5_f32, -1.0, 2.0, 0.0, -0.5, 1.5];
        let b_data = [-0.25_f32, 3.0, 0.5, -2.0, 1.0, 0.75];
        let a = device.allocate_f32(6, 64).unwrap();
        let b = device.allocate_f32(6, 64).unwrap();
        let out = device.allocate_f32(6, 64).unwrap();
        device.upload_f32(a, &a_data).unwrap();
        device.upload_f32(b, &b_data).unwrap();

        let a_value = f32_value(&[2, 3]);
        let b_value = f32_value(&[2, 3]);
        let out_value = f32_role(&[2, 3], ValueRole::Output);
        let a_ref = BufferRef {
            buffer: a,
            offset: 0,
            len: 6,
        };
        let b_ref = BufferRef {
            buffer: b,
            offset: 0,
            len: 6,
        };
        let out_ref = BufferRef {
            buffer: out,
            offset: 0,
            len: 6,
        };
        let mut result = [0.0_f32; 6];

        device
            .dispatch(
                &Op::Silu,
                &[a_ref],
                &[&a_value],
                &[out_ref],
                &[&out_value],
                &[],
            )
            .expect("silu dispatch");
        device.download_f32(out, &mut result).unwrap();
        let mut expected = a_data;
        kernels::silu(&mut expected).unwrap();
        assert_eq!(result, expected);

        device
            .dispatch(
                &Op::Mul,
                &[a_ref, b_ref],
                &[&a_value, &b_value],
                &[out_ref],
                &[&out_value],
                &[],
            )
            .expect("mul dispatch");
        device.download_f32(out, &mut result).unwrap();
        let mut expected = [0.0_f32; 6];
        kernels::mul(&a_data, &b_data, &mut expected, 6).unwrap();
        assert_eq!(result, expected);

        device
            .dispatch(
                &Op::Add,
                &[a_ref, b_ref],
                &[&a_value, &b_value],
                &[out_ref],
                &[&out_value],
                &[],
            )
            .expect("add dispatch");
        device.download_f32(out, &mut result).unwrap();
        let mut expected = [0.0_f32; 6];
        kernels::add(&a_data, &b_data, &mut expected, 6).unwrap();
        assert_eq!(result, expected);

        device
            .dispatch(
                &Op::Scale { factor: -1.5 },
                &[a_ref],
                &[&a_value],
                &[out_ref],
                &[&out_value],
                &[],
            )
            .expect("scale dispatch");
        device.download_f32(out, &mut result).unwrap();
        let mut expected = a_data;
        kernels::scale(&mut expected, -1.5).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn dispatches_softmax_rope_and_head_layouts() {
        let mut device = CpuDevice::new();
        let x_data = [0.5_f32, -1.0, 2.0, 0.0, -0.5, 1.5];
        let x = device.allocate_f32(6, 64).unwrap();
        let out = device.allocate_f32(6, 64).unwrap();
        device.upload_f32(x, &x_data).unwrap();
        let x_value = f32_value(&[2, 3]);
        let out_value = f32_role(&[2, 3], ValueRole::Output);
        let x_ref = BufferRef {
            buffer: x,
            offset: 0,
            len: 6,
        };
        let out_ref = BufferRef {
            buffer: out,
            offset: 0,
            len: 6,
        };
        let mut result = [0.0_f32; 6];

        device
            .dispatch(
                &Op::Softmax,
                &[x_ref],
                &[&x_value],
                &[out_ref],
                &[&out_value],
                &[],
            )
            .expect("softmax dispatch");
        device.download_f32(out, &mut result).unwrap();
        let mut expected = x_data;
        kernels::softmax(&mut expected, 2, 3).unwrap();
        assert_eq!(result, expected);

        device
            .dispatch(
                &Op::SoftmaxCausal,
                &[x_ref],
                &[&x_value],
                &[out_ref],
                &[&out_value],
                &[2, 1],
            )
            .expect("causal softmax dispatch");
        device.download_f32(out, &mut result).unwrap();
        let mut expected = x_data;
        kernels::softmax_causal(&mut expected, 2, 3, 2, 1).unwrap();
        assert_eq!(result, expected);

        let rope_data: [f32; 16] = core::array::from_fn(|index| (index as f32 - 7.0) * 0.125);
        let positions = [0_u32, 1];
        let rope_input = device.allocate_f32(16, 64).unwrap();
        let rope_output = device.allocate_f32(16, 64).unwrap();
        let position_buffer = device.allocate_u32(2, 64).unwrap();
        device.upload_f32(rope_input, &rope_data).unwrap();
        device.upload_u32(position_buffer, &positions).unwrap();
        let rope_value = f32_value(&[1, 2, 2, 4]);
        let position_value = u32_value(&[1, 2]);
        let rope_out_value = f32_role(&[1, 2, 2, 4], ValueRole::Output);
        let rope_ref = BufferRef {
            buffer: rope_input,
            offset: 0,
            len: 16,
        };
        let position_ref = BufferRef {
            buffer: position_buffer,
            offset: 0,
            len: 2,
        };
        let rope_out_ref = BufferRef {
            buffer: rope_output,
            offset: 0,
            len: 16,
        };
        device
            .dispatch(
                &Op::Rope {
                    variant: RopeVariant::Neox,
                    theta: 10_000.0,
                },
                &[rope_ref, position_ref],
                &[&rope_value, &position_value],
                &[rope_out_ref],
                &[&rope_out_value],
                &[],
            )
            .expect("rope dispatch");
        let mut expected = rope_data;
        kernels::rope_neox(&mut expected, &positions, 1, 2, 2, 4, 10_000.0).unwrap();
        let mut result = [0.0_f32; 16];
        device.download_f32(rope_output, &mut result).unwrap();
        assert_eq!(result, expected);

        let split_input = device.allocate_f32(16, 64).unwrap();
        let split_output = device.allocate_f32(16, 64).unwrap();
        let merge_output = device.allocate_f32(16, 64).unwrap();
        device.upload_f32(split_input, &rope_data).unwrap();
        let split_value = f32_value(&[1, 2, 8]);
        let split_out_value = f32_value(&[1, 2, 2, 4]);
        let merge_out_value = f32_role(&[1, 2, 8], ValueRole::Output);
        let split_ref = BufferRef {
            buffer: split_input,
            offset: 0,
            len: 16,
        };
        let split_out_ref = BufferRef {
            buffer: split_output,
            offset: 0,
            len: 16,
        };
        let merge_out_ref = BufferRef {
            buffer: merge_output,
            offset: 0,
            len: 16,
        };
        device
            .dispatch(
                &Op::SplitHeads {
                    heads: 2,
                    head_dim: 4,
                },
                &[split_ref],
                &[&split_value],
                &[split_out_ref],
                &[&split_out_value],
                &[],
            )
            .expect("split_heads dispatch");
        let mut expected = [0.0_f32; 16];
        kernels::split_heads(&rope_data, &mut expected, 1, 2, 2, 4).unwrap();
        let mut result = [0.0_f32; 16];
        device.download_f32(split_output, &mut result).unwrap();
        assert_eq!(result, expected);

        device
            .dispatch(
                &Op::MergeHeads {
                    heads: 2,
                    head_dim: 4,
                },
                &[split_out_ref],
                &[&split_out_value],
                &[merge_out_ref],
                &[&merge_out_value],
                &[],
            )
            .expect("merge_heads dispatch");
        device.download_f32(merge_output, &mut result).unwrap();
        assert_eq!(result, rope_data);
    }

    #[test]
    fn dispatches_matmul_nn_and_fused_ops() {
        let mut device = CpuDevice::new();
        let a_data: Vec<f32> = (0..6).map(|index| (index as f32 - 2.0) * 0.25).collect();
        let b_data: Vec<f32> = (0..12).map(|index| (index as f32 - 5.0) * 0.125).collect();
        let a = device.allocate_f32(6, 64).unwrap();
        let b = device.allocate_f32(12, 64).unwrap();
        let out = device.allocate_f32(8, 64).unwrap();
        device.upload_f32(a, &a_data).unwrap();
        device.upload_f32(b, &b_data).unwrap();
        let a_value = f32_value(&[2, 3]);
        let b_value = f32_role(&[3, 4], ValueRole::Param);
        let out_value = f32_role(&[2, 4], ValueRole::Output);
        let a_ref = BufferRef {
            buffer: a,
            offset: 0,
            len: 6,
        };
        let b_ref = BufferRef {
            buffer: b,
            offset: 0,
            len: 12,
        };
        let out_ref = BufferRef {
            buffer: out,
            offset: 0,
            len: 8,
        };
        device
            .dispatch(
                &Op::MatMulNN,
                &[a_ref, b_ref],
                &[&a_value, &b_value],
                &[out_ref],
                &[&out_value],
                &[],
            )
            .expect("matmul_nn dispatch");
        let mut expected = [0.0_f32; 8];
        kernels::matmul_nn(&a_data, &b_data, &mut expected, 1, 2, 3, 4).unwrap();
        let mut result = [0.0_f32; 8];
        device.download_f32(out, &mut result).unwrap();
        assert_eq!(result, expected);

        let x_data = [0.25_f32, -0.5, 1.0, 0.75, -0.25, 0.5];
        let gate_data: Vec<f32> = (0..12)
            .map(|index| (index as f32 % 5.0 - 2.0) * 0.3)
            .collect();
        let up_data: Vec<f32> = (0..12)
            .map(|index| (index as f32 % 3.0 - 1.0) * 0.4)
            .collect();
        let down_data: Vec<f32> = (0..12)
            .map(|index| (index as f32 % 7.0 - 3.0) * 0.2)
            .collect();
        let x = device.allocate_f32(6, 64).unwrap();
        let gate = device.allocate_f32(12, 64).unwrap();
        let up = device.allocate_f32(12, 64).unwrap();
        let down = device.allocate_f32(12, 64).unwrap();
        let swiglu_out = device.allocate_f32(6, 64).unwrap();
        device.upload_f32(x, &x_data).unwrap();
        device.upload_f32(gate, &gate_data).unwrap();
        device.upload_f32(up, &up_data).unwrap();
        device.upload_f32(down, &down_data).unwrap();
        let x_value = f32_value(&[2, 3]);
        let gate_value = f32_role(&[4, 3], ValueRole::Param);
        let up_value = f32_role(&[4, 3], ValueRole::Param);
        let down_value = f32_role(&[3, 4], ValueRole::Param);
        let swiglu_out_value = f32_role(&[2, 3], ValueRole::Output);
        let refs = [
            BufferRef {
                buffer: x,
                offset: 0,
                len: 6,
            },
            BufferRef {
                buffer: gate,
                offset: 0,
                len: 12,
            },
            BufferRef {
                buffer: up,
                offset: 0,
                len: 12,
            },
            BufferRef {
                buffer: down,
                offset: 0,
                len: 12,
            },
        ];
        let swiglu_out_ref = BufferRef {
            buffer: swiglu_out,
            offset: 0,
            len: 6,
        };
        device
            .dispatch(
                &Op::SwiGlu,
                &refs,
                &[&x_value, &gate_value, &up_value, &down_value],
                &[swiglu_out_ref],
                &[&swiglu_out_value],
                &[],
            )
            .expect("swiglu dispatch");
        let mut expected = [0.0_f32; 6];
        kernels::swiglu(
            &x_data,
            &gate_data,
            &up_data,
            &down_data,
            &mut expected,
            2,
            3,
            4,
        )
        .unwrap();
        let mut result = [0.0_f32; 6];
        device.download_f32(swiglu_out, &mut result).unwrap();
        assert_eq!(result, expected);

        let norm_data = [1.0_f32, 0.5, 1.5];
        let matrix_data: Vec<f32> = (0..12).map(|index| (index as f32 - 4.0) * 0.2).collect();
        let norm = device.allocate_f32(3, 64).unwrap();
        let matrix = device.allocate_f32(12, 64).unwrap();
        let fused_out = device.allocate_f32(8, 64).unwrap();
        let x = device.allocate_f32(6, 64).unwrap();
        device.upload_f32(x, &x_data).unwrap();
        device.upload_f32(norm, &norm_data).unwrap();
        device.upload_f32(matrix, &matrix_data).unwrap();
        let norm_value = f32_role(&[3], ValueRole::Param);
        let matrix_value = f32_role(&[4, 3], ValueRole::Param);
        let fused_out_value = f32_role(&[2, 4], ValueRole::Output);
        let x_ref = BufferRef {
            buffer: x,
            offset: 0,
            len: 6,
        };
        let norm_ref = BufferRef {
            buffer: norm,
            offset: 0,
            len: 3,
        };
        let matrix_ref = BufferRef {
            buffer: matrix,
            offset: 0,
            len: 12,
        };
        let fused_out_ref = BufferRef {
            buffer: fused_out,
            offset: 0,
            len: 8,
        };
        device
            .dispatch(
                &Op::RmsNormMatMul { eps: 1e-5 },
                &[x_ref, norm_ref, matrix_ref],
                &[&x_value, &norm_value, &matrix_value],
                &[fused_out_ref],
                &[&fused_out_value],
                &[],
            )
            .expect("rms_norm_matmul dispatch");
        let mut expected = [0.0_f32; 8];
        kernels::rms_norm_matmul(
            &x_data,
            &norm_data,
            &matrix_data,
            &mut expected,
            2,
            3,
            4,
            1e-5,
        )
        .unwrap();
        let mut result = [0.0_f32; 8];
        device.download_f32(fused_out, &mut result).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn dispatches_kv_append_and_causal_attention_with_mutable_state() {
        let mut device = CpuDevice::new();
        let q_data = [0.1_f32, -0.2, 0.3, 0.4];
        let k_data = [0.5_f32, -0.1, 0.2, 0.7];
        let v_data = [0.9_f32, -0.4, 0.6, 0.2];
        let q = device.allocate_f32(4, 64).unwrap();
        let k = device.allocate_f32(4, 64).unwrap();
        let v = device.allocate_f32(4, 64).unwrap();
        let state_append = device.allocate_f32(16, 64).unwrap();
        let state_attention = device.allocate_f32(16, 64).unwrap();
        let out = device.allocate_f32(4, 64).unwrap();
        device.upload_f32(q, &q_data).unwrap();
        device.upload_f32(k, &k_data).unwrap();
        device.upload_f32(v, &v_data).unwrap();

        let q_value = f32_value(&[1, 1, 2, 2]);
        let k_value = f32_value(&[1, 1, 2, 2]);
        let v_value = f32_value(&[1, 1, 2, 2]);
        let out_value = f32_role(&[1, 1, 2, 2], ValueRole::Output);
        let state_value = f32_role(&[1, 1, 2, 4, 1, 2], ValueRole::State);
        let q_ref = BufferRef {
            buffer: q,
            offset: 0,
            len: 4,
        };
        let k_ref = BufferRef {
            buffer: k,
            offset: 0,
            len: 4,
        };
        let v_ref = BufferRef {
            buffer: v,
            offset: 0,
            len: 4,
        };
        let state_append_ref = BufferRef {
            buffer: state_append,
            offset: 0,
            len: 16,
        };
        let state_attention_ref = BufferRef {
            buffer: state_attention,
            offset: 0,
            len: 16,
        };
        let out_ref = BufferRef {
            buffer: out,
            offset: 0,
            len: 4,
        };

        device
            .dispatch(
                &Op::KvAppend { layer: 0 },
                &[k_ref, v_ref, state_append_ref],
                &[&k_value, &v_value, &state_value],
                &[],
                &[],
                &[0],
            )
            .expect("kv_append dispatch");
        let mut expected_cache = [0.0_f32; 16];
        kernels::kv_append(
            &k_data,
            &v_data,
            &mut expected_cache,
            1,
            1,
            0,
            1,
            2,
            4,
            0,
            2,
        )
        .unwrap();
        let mut cache = [0.0_f32; 16];
        device.download_f32(state_append, &mut cache).unwrap();
        assert_eq!(cache, expected_cache);

        device
            .dispatch(
                &Op::CausalAttention {
                    layer: 0,
                    heads: 1,
                    kv_heads: 1,
                    head_dim: 2,
                    scale: 0.5,
                },
                &[q_ref, k_ref, v_ref, state_attention_ref],
                &[&q_value, &k_value, &v_value, &state_value],
                &[out_ref],
                &[&out_value],
                &[0],
            )
            .expect("causal_attention dispatch");
        let mut expected_out = [0.0_f32; 4];
        let mut expected_state = [0.0_f32; 16];
        kernels::causal_attention(
            &q_data,
            &k_data,
            &v_data,
            &mut expected_state,
            &mut expected_out,
            1,
            1,
            0,
            1,
            1,
            2,
            0,
            2,
            4,
            0.5,
        )
        .unwrap();
        let mut result = [0.0_f32; 4];
        device.download_f32(out, &mut result).unwrap();
        assert_eq!(result, expected_out);
        device.download_f32(state_attention, &mut cache).unwrap();
        assert_eq!(cache, expected_state);
    }

    #[test]
    fn rejects_overlapping_ranges() {
        let mut device = CpuDevice::new();
        let buffer = device.allocate_f32(8, 64).unwrap();
        let value = f32_value(&[4]);
        let reference = BufferRef {
            buffer,
            offset: 0,
            len: 4,
        };
        let overlapping = BufferRef {
            buffer,
            offset: 2,
            len: 4,
        };
        let output_buffer = device.allocate_f32(4, 64).unwrap();
        let output_reference = BufferRef {
            buffer: output_buffer,
            offset: 0,
            len: 4,
        };
        let error = device
            .dispatch(
                &Op::Add,
                &[reference, overlapping],
                &[&value, &value],
                &[output_reference],
                &[&value],
                &[],
            )
            .expect_err("overlapping inputs must be rejected");
        assert_eq!(
            error,
            DeviceError::Execution("one command aliases overlapping input ranges")
        );
    }
}
