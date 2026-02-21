import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from ._core_utils import validate_core_idx


@dataclass
class SimInstruction:
    opname: str
    args: Dict[str, Any] = field(default_factory=dict)
    tensors: Dict[str, Any] = field(default_factory=dict)
    tensor_dtypes: Dict[str, str] = field(default_factory=dict)
    atomic_enabled: Optional[bool] = None
    atomic_dtype: Optional[str] = None


_SIM_LOGGER = logging.getLogger("easyasc.simulator.cube")
if not _SIM_LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _SIM_LOGGER.addHandler(_handler)
_SIM_LOGGER.setLevel(logging.INFO)
_SIM_LOGGER.propagate = False


class PipeBase:
    pipe_name = ""

    def __init__(self, core_idx: int) -> None:
        self.core_idx = validate_core_idx(core_idx)
        self.instructions: List[SimInstruction] = []

    def issue(self, instruction: SimInstruction) -> None:
        if not isinstance(instruction, SimInstruction):
            raise TypeError(f"instruction must be SimInstruction, got: {type(instruction)}")
        self.instructions.append(instruction)

    def execute_instruction(self, instruction: SimInstruction) -> None:
        if instruction.opname != "sim_print":
            return
        payload = instruction.args.get("payload", [])
        if not isinstance(payload, (list, tuple)):
            raise TypeError(f"sim_print payload must be list/tuple, got: {type(payload)}")
        message = " ".join(str(item) for item in payload)
        _SIM_LOGGER.info(f"[cube][core={self.core_idx}][pipe={self.pipe_name}] {message}")

    def execute_all(self) -> None:
        for instruction in self.instructions:
            self.execute_instruction(instruction)

    def clear(self) -> None:
        self.instructions = []


class MTE2Pipe(PipeBase):
    pipe_name = "MTE2"

    @staticmethod
    def _resolve_int(value: Any, label: str) -> int:
        if isinstance(value, bool):
            return int(value)
        if not isinstance(value, (int, float)):
            raise TypeError(f"{label} must resolve to int/float, got: {type(value)}")
        return int(value)

    @staticmethod
    def _align16(value: int) -> int:
        return ((value + 15) // 16) * 16

    def _execute_gm_to_l1_nd2nz(self, instruction: SimInstruction) -> None:
        dst = instruction.tensors.get("dst")
        src = instruction.tensors.get("src")
        if not isinstance(dst, torch.Tensor):
            raise TypeError(f"gm_to_l1_nd2nz requires dst tensor view, got: {type(dst)}")
        if not isinstance(src, torch.Tensor):
            raise TypeError(f"gm_to_l1_nd2nz requires src tensor view, got: {type(src)}")
        if dst.dim() != 2:
            raise ValueError(f"gm_to_l1_nd2nz dst must be 2D, got dim={dst.dim()}")
        if src.dim() != 2:
            raise ValueError(f"gm_to_l1_nd2nz src must be 2D, got dim={src.dim()}")

        h = self._resolve_int(instruction.args.get("M"), "gm_to_l1_nd2nz M")
        w = self._resolve_int(instruction.args.get("N"), "gm_to_l1_nd2nz N")
        w_src = self._resolve_int(instruction.args.get("N_src"), "gm_to_l1_nd2nz N_src")
        h_dst = self._resolve_int(instruction.args.get("M_dst"), "gm_to_l1_nd2nz M_dst")
        if h < 0 or w < 0 or w_src < 0 or h_dst < 0:
            raise ValueError(
                "gm_to_l1_nd2nz requires non-negative M/N/N_src/M_dst, got: "
                f"M={h}, N={w}, N_src={w_src}, M_dst={h_dst}"
            )
        if h > int(src.shape[0]):
            raise ValueError(f"gm_to_l1_nd2nz M={h} exceeds src rows={int(src.shape[0])}")
        if w > int(src.shape[1]):
            raise ValueError(f"gm_to_l1_nd2nz N={w} exceeds src cols={int(src.shape[1])}")
        if w > w_src:
            raise ValueError(f"gm_to_l1_nd2nz N={w} cannot exceed N_src={w_src}")
        if h > h_dst:
            raise ValueError(f"gm_to_l1_nd2nz M={h} cannot exceed M_dst={h_dst}")

        elem_size = int(dst.element_size())
        if elem_size <= 0 or (32 % elem_size) != 0:
            raise ValueError(
                "gm_to_l1_nd2nz unsupported dst element size for C0=32-byte layout: "
                f"{elem_size} bytes"
            )
        c0 = 32 // elem_size
        dst_stride_h = self._align16(h_dst)
        needed = ((w + c0 - 1) // c0) * dst_stride_h * c0
        dst_flat = dst.reshape(-1)
        if needed > int(dst_flat.numel()):
            raise MemoryError(
                "gm_to_l1_nd2nz destination view is too small for NZ layout: "
                f"need={needed}, have={int(dst_flat.numel())}, "
                f"M_dst={h_dst}, N={w}, C0={c0}, aligned_M_dst={dst_stride_h}"
            )

        dst_flat.zero_()
        for row in range(h):
            for col in range(w):
                block = col // c0
                inner = col % c0
                dst_idx = block * dst_stride_h * c0 + row * c0 + inner
                dst_flat[dst_idx] = src[row, col]

    def execute_instruction(self, instruction: SimInstruction) -> None:
        if instruction.opname == "gm_to_l1_nd2nz":
            self._execute_gm_to_l1_nd2nz(instruction)
            return
        super().execute_instruction(instruction)


class MTE1Pipe(PipeBase):
    pipe_name = "MTE1"


class MPipe(PipeBase):
    pipe_name = "M"


class FIXPipe(PipeBase):
    pipe_name = "FIX"
