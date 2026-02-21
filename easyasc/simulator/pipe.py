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
    seq: int = -1


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
        if h == 0 or w == 0:
            return

        block_count = (w + c0 - 1) // c0
        src_hw = src[:h, :w]
        dst_panel = dst_flat[: block_count * dst_stride_h * c0].reshape(block_count, dst_stride_h, c0)

        full_blocks = w // c0
        if full_blocks > 0:
            src_full = src_hw[:, : full_blocks * c0].reshape(h, full_blocks, c0).permute(1, 0, 2)
            dst_panel[:full_blocks, :h, :].copy_(src_full)

        tail = w - full_blocks * c0
        if tail > 0:
            start_col = full_blocks * c0
            dst_panel[full_blocks, :h, :tail].copy_(src_hw[:, start_col : start_col + tail])

    def execute_instruction(self, instruction: SimInstruction) -> None:
        if instruction.opname == "gm_to_l1_nd2nz":
            self._execute_gm_to_l1_nd2nz(instruction)
            return
        super().execute_instruction(instruction)


class MTE1Pipe(PipeBase):
    pipe_name = "MTE1"

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

    @staticmethod
    def _align_to(value: int, align: int) -> int:
        if align <= 0:
            raise ValueError(f"align must be positive, got: {align}")
        return ((value + align - 1) // align) * align

    @staticmethod
    def _decode_nz_to_nd(
        src_flat: torch.Tensor,
        m_rows: int,
        n_cols: int,
        src_stride_m: int,
        c0: int,
    ) -> torch.Tensor:
        out = src_flat.new_zeros((m_rows, n_cols))
        if m_rows == 0 or n_cols == 0:
            return out
        blocks = (n_cols + c0 - 1) // c0
        src_panel = src_flat[: blocks * src_stride_m * c0].reshape(blocks, src_stride_m, c0)
        full_blocks = n_cols // c0
        if full_blocks > 0:
            src_full = src_panel[:full_blocks, :m_rows, :]
            out[:, : full_blocks * c0].copy_(src_full.permute(1, 0, 2).reshape(m_rows, full_blocks * c0))
        tail = n_cols - full_blocks * c0
        if tail > 0:
            start_col = full_blocks * c0
            out[:, start_col : start_col + tail].copy_(src_panel[full_blocks, :m_rows, :tail])
        return out

    def _execute_l1_to_l0(self, instruction: SimInstruction) -> None:
        dst = instruction.tensors.get("dst")
        src = instruction.tensors.get("src")
        if not isinstance(dst, torch.Tensor):
            raise TypeError(f"l1_to_l0 requires dst tensor view, got: {type(dst)}")
        if not isinstance(src, torch.Tensor):
            raise TypeError(f"l1_to_l0 requires src tensor view, got: {type(src)}")
        if dst.dim() != 2:
            raise ValueError(f"l1_to_l0 dst must be 2D, got dim={dst.dim()}")
        if src.dim() != 2:
            raise ValueError(f"l1_to_l0 src must be 2D, got dim={src.dim()}")

        m_dst = self._resolve_int(instruction.args.get("m_dst"), "l1_to_l0 m_dst")
        n_dst = self._resolve_int(instruction.args.get("n_dst"), "l1_to_l0 n_dst")
        m_src = self._resolve_int(instruction.args.get("m_src"), "l1_to_l0 m_src")
        n_src = self._resolve_int(instruction.args.get("n_src"), "l1_to_l0 n_src")
        src_is_transpose = bool(instruction.args.get("src_is_transpose", False))
        if m_dst < 0 or n_dst < 0 or m_src < 0 or n_src < 0:
            raise ValueError(
                "l1_to_l0 requires non-negative m_dst/n_dst/m_src/n_src, got: "
                f"m_dst={m_dst}, n_dst={n_dst}, m_src={m_src}, n_src={n_src}"
            )
        if m_dst > m_src:
            raise ValueError(f"l1_to_l0 m_dst={m_dst} cannot exceed m_src={m_src}")
        if n_dst > n_src:
            raise ValueError(f"l1_to_l0 n_dst={n_dst} cannot exceed n_src={n_src}")

        elem_size = int(dst.element_size())
        if elem_size <= 0 or (32 % elem_size) != 0:
            raise ValueError(
                "l1_to_l0 unsupported dst element size for C0=32-byte layout: "
                f"{elem_size} bytes"
            )
        c0 = 32 // elem_size

        src_stride_m = self._align16(m_src)
        src_needed = ((n_src + c0 - 1) // c0) * src_stride_m * c0
        src_flat = src.reshape(-1)
        if src_needed > int(src_flat.numel()):
            raise MemoryError(
                "l1_to_l0 source view is too small for NZ layout: "
                f"need={src_needed}, have={int(src_flat.numel())}, "
                f"m_src={m_src}, n_src={n_src}, c0={c0}, aligned_m_src={src_stride_m}"
            )

        dst_flat = dst.reshape(-1)
        if not src_is_transpose:
            dst_stride_m = self._align16(m_dst)
            dst_needed = ((n_dst + c0 - 1) // c0) * dst_stride_m * c0
            if dst_needed > int(dst_flat.numel()):
                raise MemoryError(
                    "l1_to_l0 dst view is too small for NZ layout: "
                    f"need={dst_needed}, have={int(dst_flat.numel())}, "
                    f"m_dst={m_dst}, n_dst={n_dst}, c0={c0}, aligned_m_dst={dst_stride_m}"
                )
            dst_flat.zero_()
            if m_dst == 0 or n_dst == 0:
                return
            blocks = (n_dst + c0 - 1) // c0
            src_panel = src_flat[: blocks * src_stride_m * c0].reshape(blocks, src_stride_m, c0)
            dst_panel = dst_flat[: blocks * dst_stride_m * c0].reshape(blocks, dst_stride_m, c0)
            full_blocks = n_dst // c0
            if full_blocks > 0:
                dst_panel[:full_blocks, :m_dst, :].copy_(src_panel[:full_blocks, :m_dst, :])
            tail = n_dst - full_blocks * c0
            if tail > 0:
                dst_panel[full_blocks, :m_dst, :tail].copy_(src_panel[full_blocks, :m_dst, :tail])
            return

        # NZ -> ZN transpose path: int8 uses 32-row tile, others use 16-row tile.
        row_block = 32 if elem_size == 1 else 16
        n_align = self._align_to(n_dst, 32 if elem_size == 1 else c0)
        m_blocks = (m_dst + row_block - 1) // row_block
        dst_needed = m_blocks * row_block * n_align
        if dst_needed > int(dst_flat.numel()):
            raise MemoryError(
                "l1_to_l0 dst view is too small for ZN layout: "
                f"need={dst_needed}, have={int(dst_flat.numel())}, "
                f"m_dst={m_dst}, n_dst={n_dst}, row_block={row_block}, n_align={n_align}"
            )
        dst_flat.zero_()
        if m_dst == 0 or n_dst == 0:
            return

        logical = self._decode_nz_to_nd(src_flat, m_dst, n_dst, src_stride_m, c0)
        logical_pad = logical.new_zeros((m_blocks * row_block, n_align))
        logical_pad[:m_dst, :n_dst].copy_(logical)
        dst_zn = dst_flat[:dst_needed].reshape(m_blocks, n_align, row_block)
        dst_zn.copy_(logical_pad.reshape(m_blocks, row_block, n_align).permute(0, 2, 1))

    def execute_instruction(self, instruction: SimInstruction) -> None:
        if instruction.opname == "l1_to_l0":
            self._execute_l1_to_l0(instruction)
            return
        super().execute_instruction(instruction)


class MPipe(PipeBase):
    pipe_name = "M"

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

    @staticmethod
    def _decode_nz_to_nd(
        src_flat: torch.Tensor,
        m_rows: int,
        n_cols: int,
        src_stride_m: int,
        c0: int,
    ) -> torch.Tensor:
        out = src_flat.new_zeros((m_rows, n_cols))
        if m_rows == 0 or n_cols == 0:
            return out
        blocks = (n_cols + c0 - 1) // c0
        src_panel = src_flat[: blocks * src_stride_m * c0].reshape(blocks, src_stride_m, c0)
        full_blocks = n_cols // c0
        if full_blocks > 0:
            src_full = src_panel[:full_blocks, :m_rows, :]
            out[:, : full_blocks * c0].copy_(src_full.permute(1, 0, 2).reshape(m_rows, full_blocks * c0))
        tail = n_cols - full_blocks * c0
        if tail > 0:
            start_col = full_blocks * c0
            out[:, start_col : start_col + tail].copy_(src_panel[full_blocks, :m_rows, :tail])
        return out

    @staticmethod
    def _encode_nd_to_nz(
        logical: torch.Tensor,
        dst_flat: torch.Tensor,
        m_rows: int,
        n_cols: int,
        dst_stride_m: int,
        c0: int,
    ) -> None:
        if m_rows == 0 or n_cols == 0:
            return
        blocks = (n_cols + c0 - 1) // c0
        dst_panel = dst_flat[: blocks * dst_stride_m * c0].reshape(blocks, dst_stride_m, c0)
        full_blocks = n_cols // c0
        if full_blocks > 0:
            src_full = logical[:, : full_blocks * c0].reshape(m_rows, full_blocks, c0).permute(1, 0, 2)
            dst_panel[:full_blocks, :m_rows, :].copy_(src_full)
        tail = n_cols - full_blocks * c0
        if tail > 0:
            start_col = full_blocks * c0
            dst_panel[full_blocks, :m_rows, :tail].copy_(logical[:, start_col : start_col + tail])

    @staticmethod
    def _choose_compute_dtype(dst_dtype: torch.dtype) -> torch.dtype:
        if dst_dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        if dst_dtype in (torch.int8, torch.uint8, torch.int16):
            return torch.int32
        return dst_dtype

    def _execute_mmad(self, instruction: SimInstruction) -> None:
        dst = instruction.tensors.get("dst")
        src_a = instruction.tensors.get("src_a")
        src_b = instruction.tensors.get("src_b")
        if not isinstance(dst, torch.Tensor):
            raise TypeError(f"mmad requires dst tensor view, got: {type(dst)}")
        if not isinstance(src_a, torch.Tensor):
            raise TypeError(f"mmad requires src_a tensor view, got: {type(src_a)}")
        if not isinstance(src_b, torch.Tensor):
            raise TypeError(f"mmad requires src_b tensor view, got: {type(src_b)}")
        if dst.dim() != 2:
            raise ValueError(f"mmad dst must be 2D, got dim={dst.dim()}")
        if src_a.dim() != 2:
            raise ValueError(f"mmad src_a must be 2D, got dim={src_a.dim()}")
        if src_b.dim() != 2:
            raise ValueError(f"mmad src_b must be 2D, got dim={src_b.dim()}")

        m = self._resolve_int(instruction.args.get("M"), "mmad M")
        n = self._resolve_int(instruction.args.get("N"), "mmad N")
        k = self._resolve_int(instruction.args.get("K"), "mmad K")
        is_init = bool(instruction.args.get("is_init", True))
        if m < 0 or n < 0 or k < 0:
            raise ValueError(f"mmad requires non-negative M/N/K, got: M={m}, N={n}, K={k}")

        src_a_rows = int(src_a.shape[0])
        src_a_cols = int(src_a.shape[1])
        src_b_rows = int(src_b.shape[0])
        src_b_cols = int(src_b.shape[1])
        dst_rows = int(dst.shape[0])
        dst_cols = int(dst.shape[1])
        if m > src_a_rows:
            raise ValueError(f"mmad M={m} exceeds src_a rows={src_a_rows}")
        if n > src_b_rows:
            raise ValueError(f"mmad N={n} exceeds src_b rows={src_b_rows}")
        if m > dst_rows:
            raise ValueError(f"mmad M={m} exceeds dst rows={dst_rows}")
        if n > dst_cols:
            raise ValueError(f"mmad N={n} exceeds dst cols={dst_cols}")
        if k > src_a_cols:
            raise ValueError(f"mmad K={k} exceeds src_a cols={src_a_cols}")
        if k > src_b_cols:
            raise ValueError(f"mmad K={k} exceeds src_b cols={src_b_cols}")

        elem_a = int(src_a.element_size())
        elem_b = int(src_b.element_size())
        elem_dst = int(dst.element_size())
        if elem_a <= 0 or (32 % elem_a) != 0:
            raise ValueError(f"mmad unsupported src_a element size: {elem_a}")
        if elem_b <= 0 or (32 % elem_b) != 0:
            raise ValueError(f"mmad unsupported src_b element size: {elem_b}")
        if elem_dst <= 0 or (32 % elem_dst) != 0:
            raise ValueError(f"mmad unsupported dst element size: {elem_dst}")
        c0_a = 32 // elem_a
        c0_b = 32 // elem_b
        c0_dst = 32 // elem_dst

        src_a_stride_m = self._align16(src_a_rows)
        src_b_stride_m = self._align16(src_b_rows)
        dst_stride_m = self._align16(dst_rows)
        need_src_a = ((k + c0_a - 1) // c0_a) * src_a_stride_m * c0_a
        need_src_b = ((k + c0_b - 1) // c0_b) * src_b_stride_m * c0_b
        need_dst = ((n + c0_dst - 1) // c0_dst) * dst_stride_m * c0_dst

        src_a_flat = src_a.reshape(-1)
        src_b_flat = src_b.reshape(-1)
        dst_flat = dst.reshape(-1)
        if need_src_a > int(src_a_flat.numel()):
            raise MemoryError(
                "mmad src_a view is too small for NZ layout: "
                f"need={need_src_a}, have={int(src_a_flat.numel())}"
            )
        if need_src_b > int(src_b_flat.numel()):
            raise MemoryError(
                "mmad src_b view is too small for NZ layout: "
                f"need={need_src_b}, have={int(src_b_flat.numel())}"
            )
        if need_dst > int(dst_flat.numel()):
            raise MemoryError(
                "mmad dst view is too small for NZ layout: "
                f"need={need_dst}, have={int(dst_flat.numel())}"
            )

        a_nd = self._decode_nz_to_nd(src_a_flat, m, k, src_a_stride_m, c0_a)
        b_nd = self._decode_nz_to_nd(src_b_flat, n, k, src_b_stride_m, c0_b)

        compute_dtype = self._choose_compute_dtype(dst.dtype)
        prod = torch.matmul(a_nd.to(compute_dtype), b_nd.to(compute_dtype).transpose(0, 1))
        if not is_init:
            prev = self._decode_nz_to_nd(dst_flat, m, n, dst_stride_m, c0_dst).to(compute_dtype)
            prod = prev + prod
        out = prod.to(dst.dtype)
        self._encode_nd_to_nz(out, dst_flat, m, n, dst_stride_m, c0_dst)

    def execute_instruction(self, instruction: SimInstruction) -> None:
        if instruction.opname == "mmad":
            self._execute_mmad(instruction)
            return
        super().execute_instruction(instruction)


class FIXPipe(PipeBase):
    pipe_name = "FIX"

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

    @staticmethod
    def _decode_nz_to_nd(
        src_flat: torch.Tensor,
        m_rows: int,
        n_cols: int,
        src_stride_m: int,
        c0: int,
    ) -> torch.Tensor:
        out = src_flat.new_zeros((m_rows, n_cols))
        if m_rows == 0 or n_cols == 0:
            return out
        blocks = (n_cols + c0 - 1) // c0
        src_panel = src_flat[: blocks * src_stride_m * c0].reshape(blocks, src_stride_m, c0)
        full_blocks = n_cols // c0
        if full_blocks > 0:
            src_full = src_panel[:full_blocks, :m_rows, :]
            out[:, : full_blocks * c0].copy_(src_full.permute(1, 0, 2).reshape(m_rows, full_blocks * c0))
        tail = n_cols - full_blocks * c0
        if tail > 0:
            start_col = full_blocks * c0
            out[:, start_col : start_col + tail].copy_(src_panel[full_blocks, :m_rows, :tail])
        return out

    @staticmethod
    def _choose_atomic_compute_dtype(dtype: torch.dtype) -> torch.dtype:
        if dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        if dtype in (torch.int8, torch.uint8, torch.int16):
            return torch.int32
        return dtype

    def _execute_l0c_to_gm_nz2nd(self, instruction: SimInstruction) -> None:
        dst = instruction.tensors.get("dst")
        src = instruction.tensors.get("src")
        if not isinstance(dst, torch.Tensor):
            raise TypeError(f"l0c_to_gm_nz2nd requires dst tensor view, got: {type(dst)}")
        if not isinstance(src, torch.Tensor):
            raise TypeError(f"l0c_to_gm_nz2nd requires src tensor view, got: {type(src)}")
        if dst.dim() != 2:
            raise ValueError(f"l0c_to_gm_nz2nd dst must be 2D, got dim={dst.dim()}")
        if src.dim() != 2:
            raise ValueError(f"l0c_to_gm_nz2nd src must be 2D, got dim={src.dim()}")

        m = self._resolve_int(instruction.args.get("M"), "l0c_to_gm_nz2nd M")
        n = self._resolve_int(instruction.args.get("N"), "l0c_to_gm_nz2nd N")
        n_dst = self._resolve_int(instruction.args.get("N_dst"), "l0c_to_gm_nz2nd N_dst")
        m_src = self._resolve_int(instruction.args.get("M_src"), "l0c_to_gm_nz2nd M_src")
        if m < 0 or n < 0 or n_dst < 0 or m_src < 0:
            raise ValueError(
                "l0c_to_gm_nz2nd requires non-negative M/N/N_dst/M_src, got: "
                f"M={m}, N={n}, N_dst={n_dst}, M_src={m_src}"
            )
        if m > m_src:
            raise ValueError(f"l0c_to_gm_nz2nd M={m} cannot exceed M_src={m_src}")
        if n > n_dst:
            raise ValueError(f"l0c_to_gm_nz2nd N={n} cannot exceed N_dst={n_dst}")
        if m > int(src.shape[0]):
            raise ValueError(f"l0c_to_gm_nz2nd M={m} exceeds src rows={int(src.shape[0])}")
        if n > int(src.shape[1]):
            raise ValueError(f"l0c_to_gm_nz2nd N={n} exceeds src cols={int(src.shape[1])}")
        if m > int(dst.shape[0]):
            raise ValueError(f"l0c_to_gm_nz2nd M={m} exceeds dst rows={int(dst.shape[0])}")
        if n > int(dst.shape[1]):
            raise ValueError(f"l0c_to_gm_nz2nd N={n} exceeds dst cols={int(dst.shape[1])}")

        elem_size = int(src.element_size())
        if elem_size <= 0 or (32 % elem_size) != 0:
            raise ValueError(
                "l0c_to_gm_nz2nd unsupported src element size for C0=32-byte layout: "
                f"{elem_size} bytes"
            )
        c0 = 32 // elem_size
        src_stride_m = self._align16(m_src)
        src_needed = ((n + c0 - 1) // c0) * src_stride_m * c0
        src_flat = src.reshape(-1)
        if src_needed > int(src_flat.numel()):
            raise MemoryError(
                "l0c_to_gm_nz2nd source view is too small for NZ layout: "
                f"need={src_needed}, have={int(src_flat.numel())}, "
                f"M_src={m_src}, N={n}, C0={c0}, aligned_M_src={src_stride_m}"
            )

        logical = self._decode_nz_to_nd(src_flat, m, n, src_stride_m, c0).to(dst.dtype)
        if m == 0 or n == 0:
            return

        dst_region = dst[:m, :n]
        atomic_enabled = bool(instruction.atomic_enabled)
        if atomic_enabled:
            dst_dtype = instruction.tensor_dtypes.get("dst")
            atomic_dtype = instruction.atomic_dtype
            if (
                isinstance(dst_dtype, str)
                and isinstance(atomic_dtype, str)
                and dst_dtype != atomic_dtype
            ):
                _SIM_LOGGER.warning(
                    f"[cube][core={self.core_idx}][pipe={self.pipe_name}] "
                    f"atomic dtype mismatch for l0c_to_gm_nz2nd: dst={dst_dtype}, atomic={atomic_dtype}"
                )
            compute_dtype = self._choose_atomic_compute_dtype(dst_region.dtype)
            dst_region.copy_(dst_region.to(compute_dtype).add(logical.to(compute_dtype)).to(dst_region.dtype))
            return

        dst_region.copy_(logical)

    def execute_instruction(self, instruction: SimInstruction) -> None:
        if instruction.opname == "l0c_to_gm_nz2nd":
            self._execute_l0c_to_gm_nz2nd(instruction)
            return
        super().execute_instruction(instruction)
