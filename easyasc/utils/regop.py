from __future__ import annotations

from typing import Optional, Sequence, TYPE_CHECKING, Union

from .. import globvars
from .castconfig import CastConfig
from .comparemode import CompareModeType
from .datatype import DataTypeValue
from .var import Var

if TYPE_CHECKING:
    from .Tensor import Tensor
    from .reg import Reg, MaskReg


Scalar = Union[int, float, Var]


def _require_micro():
    micro = globvars.active_micro
    if micro is None:
        raise RuntimeError("RegOP只能在MicroModule中使用")
    return micro


def _default_mask(dtype: DataTypeValue) -> "MaskReg":
    micro = _require_micro()
    return micro.get_mask(dtype)


def _release_temp_reg(reg: object) -> None:
    from .reg import Reg

    micro = globvars.active_micro
    if micro is None:
        return
    if isinstance(reg, Reg) and reg.name.startswith("_tmp_reg_"):
        micro.release_reg(reg)


class RegOP:
    def __init__(self, opname: str, *inputs: object) -> None:
        self.inputs: Sequence[object] = inputs
        self.opname: str = opname
        self._mask: Optional["MaskReg"] = None

    def __repr__(self) -> str:
        return f"RegOP(opname={self.opname!r}, inputs={self.inputs!r})"

    def setmask(self, mask: "MaskReg") -> None:
        self._mask = mask

    @property
    def mask(self) -> Optional["MaskReg"]:
        return self._mask

    def release_inputs(self) -> None:
        for value in self.inputs:
            _release_temp_reg(value)

    def _output_dtype(self) -> DataTypeValue:
        from .reg import Reg

        if self.opname == "cast":
            if len(self.inputs) >= 3 and isinstance(self.inputs[2], DataTypeValue):
                return self.inputs[2]
        if not self.inputs or not isinstance(self.inputs[0], Reg):
            raise TypeError("无法推断RegOP输出dtype")
        return self.inputs[0].dtype

    def emit(self, dst: object) -> None:
        from .reg import Reg, MaskReg
        from ..stub_functions import micro

        op = self.opname
        mask = self._mask

        if op in ("add", "sub", "mul", "div", "vmax", "vmin", "vand", "vor", "vxor", "prelu"):
            src1, src2 = self.inputs[:2]
            micro_func = getattr(micro, op)
            micro_func(dst, src1, src2, mask=mask)
            return

        if op in ("exp", "abs", "relu", "sqrt", "ln", "log", "log2", "log10", "neg", "vnot", "vcopy"):
            src = self.inputs[0]
            micro_func = getattr(micro, op)
            micro_func(dst, src, mask=mask)
            return

        if op in ("vmaxs", "vmins", "adds", "muls", "lrelu", "shiftls", "shiftrs", "axpy"):
            src, value = self.inputs[:2]
            micro_func = getattr(micro, op)
            micro_func(dst, src, value, mask=mask)
            return

        if op in ("cmax", "cgmax", "cmin", "cgmin", "cadd", "cgadd", "cpadd"):
            src = self.inputs[0]
            micro_func = getattr(micro, op)
            micro_func(dst, src, mask=mask)
            return

        if op == "dup":
            src = self.inputs[0]
            micro.dup(dst, src, mask=mask)
            return

        if op == "arange":
            start = self.inputs[0]
            increase = True
            if len(self.inputs) >= 2:
                increase = bool(self.inputs[1])
            micro.arange(dst, start, increase=increase)
            return

        if op == "cast":
            src = self.inputs[0]
            cfg = self.inputs[1]
            if not isinstance(cfg, CastConfig):
                raise TypeError(f"cast需要CastConfig，当前类型: {type(cfg)}")
            if mask is None:
                if not isinstance(dst, Reg):
                    raise TypeError("cast目标必须是Reg")
                mask = _default_mask(dst.dtype)
            micro.cast(dst, src, cfg, mask)
            return

        if op == "compare":
            src1, src2, mode = self.inputs[:3]
            if not isinstance(mode, CompareModeType):
                raise TypeError(f"compare需要CompareModeType，当前类型: {type(mode)}")
            micro.compare(dst, src1, src2, mode, mask=mask)
            return

        if op == "select":
            src1, src2 = self.inputs[:2]
            micro.select(dst, src1, src2, mask=mask)
            return

        if op == "mask_not":
            src = self.inputs[0]
            micro.mask_not(dst, src, mask=mask)
            return

        if op in ("mask_and", "mask_or", "mask_xor"):
            src1, src2 = self.inputs[:2]
            micro_func = getattr(micro, op)
            micro_func(dst, src1, src2, mask=mask)
            return

        if op == "mask_mov":
            src = self.inputs[0]
            micro.mask_mov(dst, src, mask=mask)
            return

        if op == "mask_sel":
            src1, src2 = self.inputs[:2]
            micro.mask_sel(dst, src1, src2, mask=mask)
            return

        if op in ("mask_pack", "mask_unpack"):
            src = self.inputs[0]
            low_part = True
            if len(self.inputs) >= 2:
                low_part = bool(self.inputs[1])
            micro_func = getattr(micro, op)
            micro_func(dst, src, low_part=low_part)
            return

        if op in ("mask_interleave", "mask_deinterleave"):
            src0, src1, dst1 = self.inputs[:3]
            micro_func = getattr(micro, op)
            micro_func(dst, dst1, src0, src1)
            return

        if op == "move_mask_spr":
            micro.move_mask_spr(dst)
            return

        if op == "update_mask":
            cnt = self.inputs[0]
            micro.update_mask(dst, cnt)
            return

        if op in ("interleave", "deinterleave"):
            src0, src1, dst1 = self.inputs[:3]
            micro_func = getattr(micro, op)
            micro_func(dst, dst1, src0, src1)
            return

        if op == "ub_to_reg":
            src = self.inputs[0]
            blk_stride = 1 if len(self.inputs) < 2 else self.inputs[1]
            micro.ub_to_reg(dst, src, blk_stride=blk_stride, mask=mask)
            return

        if op == "reg_to_ub":
            src = self.inputs[0]
            blk_stride = 1 if len(self.inputs) < 2 else self.inputs[1]
            micro.reg_to_ub(dst, src, blk_stride=blk_stride, mask=mask)
            return

        if op == "ub_to_reg_continuous":
            src, loaddist = self.inputs[:2]
            micro.ub_to_reg_continuous(dst, src, loaddist)
            return

        if op == "reg_to_ub_continuous":
            src, storedist = self.inputs[:2]
            micro.reg_to_ub_continuous(dst, src, mask, storedist)
            return

        if op in ("reg_to_ub_downsample", "reg_to_ub_pack4", "reg_to_ub_single"):
            src = self.inputs[0]
            micro_func = getattr(micro, op)
            micro_func(dst, src, mask=mask)
            return

        if op in (
            "ub_to_reg_single",
            "ub_to_reg_upsample",
            "ub_to_reg_downsample",
            "ub_to_reg_unpack",
            "ub_to_reg_unpack4",
            "ub_to_reg_brcb",
        ):
            src = self.inputs[0]
            micro_func = getattr(micro, op)
            micro_func(dst, src)
            return

        if op == "ub_to_reg_gather":
            src, index = self.inputs[:2]
            micro.ub_to_reg_gather(dst, src, index, mask=mask)
            return

        if op == "reg_to_ub_scatter":
            src, index = self.inputs[:2]
            micro.reg_to_ub_scatter(dst, src, index, mask=mask)
            return

        if op == "gather":
            src, index = self.inputs[:2]
            micro.gather(dst, src, index)
            return

        if op == "gather_mask":
            src = self.inputs[0]
            micro.gather_mask(dst, src, mask=mask)
            return

        raise ValueError(f"Unsupported RegOP: {op}")

    def run_regop(self) -> "Reg":
        from .reg import Reg

        if self.opname in (
            "compare",
            "mask_not",
            "mask_and",
            "mask_or",
            "mask_xor",
            "mask_mov",
            "mask_sel",
            "mask_pack",
            "mask_unpack",
            "mask_interleave",
            "mask_deinterleave",
            "move_mask_spr",
            "update_mask",
            "reg_to_ub",
            "reg_to_ub_continuous",
            "reg_to_ub_downsample",
            "reg_to_ub_pack4",
            "reg_to_ub_single",
            "reg_to_ub_scatter",
        ):
            raise ValueError("该RegOP不支持直接生成Reg结果")

        self.release_inputs()
        dtype = self._output_dtype()
        micro = _require_micro()
        dst = micro.get_reg(dtype)
        self.emit(dst)
        return dst

    def astype(self, dtype: DataTypeValue, cfg: Optional[CastConfig] = None) -> "RegOP":
        if cfg is None:
            cfg = CastConfig()
        dst = self.run_regop()
        return RegOP("cast", dst, cfg, dtype)

    def exp(self) -> "RegOP":
        dst = self.run_regop()
        return RegOP("exp", dst)

    def abs(self) -> "RegOP":
        dst = self.run_regop()
        return RegOP("abs", dst)

    def sqrt(self) -> "RegOP":
        dst = self.run_regop()
        return RegOP("sqrt", dst)

    def relu(self) -> "RegOP":
        dst = self.run_regop()
        return RegOP("relu", dst)

    def ln(self) -> "RegOP":
        dst = self.run_regop()
        return RegOP("ln", dst)

    def log(self) -> "RegOP":
        dst = self.run_regop()
        return RegOP("log", dst)

    def log2(self) -> "RegOP":
        dst = self.run_regop()
        return RegOP("log2", dst)

    def log10(self) -> "RegOP":
        dst = self.run_regop()
        return RegOP("log10", dst)

    def neg(self) -> "RegOP":
        dst = self.run_regop()
        return RegOP("neg", dst)

    def vnot(self) -> "RegOP":
        dst = self.run_regop()
        return RegOP("vnot", dst)

    def vcopy(self) -> "RegOP":
        dst = self.run_regop()
        return RegOP("vcopy", dst)

    def shiftls(self, value: Scalar) -> "RegOP":
        dst = self.run_regop()
        return RegOP("shiftls", dst, value)

    def shiftrs(self, value: Scalar) -> "RegOP":
        dst = self.run_regop()
        return RegOP("shiftrs", dst, value)

    def axpy(self, value: Scalar) -> "RegOP":
        dst = self.run_regop()
        return RegOP("axpy", dst, value)

    def lrelu(self, value: Scalar) -> "RegOP":
        dst = self.run_regop()
        return RegOP("lrelu", dst, value)

    def vand(self, other: "Reg") -> "RegOP":
        dst = self.run_regop()
        return RegOP("vand", dst, other)

    def vor(self, other: "Reg") -> "RegOP":
        dst = self.run_regop()
        return RegOP("vor", dst, other)

    def vxor(self, other: "Reg") -> "RegOP":
        dst = self.run_regop()
        return RegOP("vxor", dst, other)

    def prelu(self, other: "Reg") -> "RegOP":
        dst = self.run_regop()
        return RegOP("prelu", dst, other)

    def vmax(self, other: "Reg") -> "RegOP":
        dst = self.run_regop()
        return RegOP("vmax", dst, other)

    def vmin(self, other: "Reg") -> "RegOP":
        dst = self.run_regop()
        return RegOP("vmin", dst, other)

    def cadd(self) -> "RegOP":
        dst = self.run_regop()
        return RegOP("cadd", dst)

    def cmax(self) -> "RegOP":
        dst = self.run_regop()
        return RegOP("cmax", dst)

    def cmin(self) -> "RegOP":
        dst = self.run_regop()
        return RegOP("cmin", dst)

    def cgadd(self) -> "RegOP":
        dst = self.run_regop()
        return RegOP("cgadd", dst)

    def cgmax(self) -> "RegOP":
        dst = self.run_regop()
        return RegOP("cgmax", dst)

    def cgmin(self) -> "RegOP":
        dst = self.run_regop()
        return RegOP("cgmin", dst)

    def cpadd(self) -> "RegOP":
        dst = self.run_regop()
        return RegOP("cpadd", dst)

    def dup(self) -> "RegOP":
        dst = self.run_regop()
        return RegOP("dup", dst)

    def __add__(self, other: Union[int, float, Var, "Reg", "RegOP"]) -> "RegOP":
        from .reg import Reg

        if isinstance(other, RegOP):
            src1 = self.run_regop()
            src2 = other.run_regop()
            return RegOP("add", src1, src2)
        if isinstance(other, Reg):
            src1 = self.run_regop()
            return RegOP("add", src1, other)
        if isinstance(other, (int, float, Var)):
            src1 = self.run_regop()
            return RegOP("adds", src1, other)
        raise TypeError(f"RegOP不支持与{type(other)}相加")

    def __sub__(self, other: Union[int, float, Var, "Reg", "RegOP"]) -> "RegOP":
        from .reg import Reg

        if isinstance(other, RegOP):
            src1 = self.run_regop()
            src2 = other.run_regop()
            return RegOP("sub", src1, src2)
        if isinstance(other, Reg):
            src1 = self.run_regop()
            return RegOP("sub", src1, other)
        if isinstance(other, (int, float, Var)):
            src1 = self.run_regop()
            return RegOP("adds", src1, -1 * other)
        raise TypeError(f"RegOP不支持与{type(other)}相减")

    def __mul__(self, other: Union[int, float, Var, "Reg", "RegOP", "MaskReg"]) -> "RegOP":
        from .reg import Reg, MaskReg

        if isinstance(other, MaskReg):
            self._mask = other
            return self
        if isinstance(other, RegOP):
            src1 = self.run_regop()
            src2 = other.run_regop()
            return RegOP("mul", src1, src2)
        if isinstance(other, Reg):
            src1 = self.run_regop()
            return RegOP("mul", src1, other)
        if isinstance(other, (int, float, Var)):
            src1 = self.run_regop()
            return RegOP("muls", src1, other)
        raise TypeError(f"RegOP不支持与{type(other)}相乘")

    def __truediv__(self, other: Union[int, float, Var, "Reg", "RegOP"]) -> "RegOP":
        from .reg import Reg

        if isinstance(other, RegOP):
            src1 = self.run_regop()
            src2 = other.run_regop()
            return RegOP("div", src1, src2)
        if isinstance(other, Reg):
            src1 = self.run_regop()
            return RegOP("div", src1, other)
        if isinstance(other, (int, float, Var)):
            src1 = self.run_regop()
            return RegOP("muls", src1, 1 / other)
        raise TypeError(f"RegOP不支持与{type(other)}相除")
