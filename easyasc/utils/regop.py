from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING, Union

from .. import globvars
from .castconfig import CastConfig
from .comparemode import CompareModeType
from .datatype import DataTypeValue
from .var import Var

if TYPE_CHECKING:
    from ..micro.micromodule import MicroModule
    from ..stub_functions.micro.datamove import LoadDistValue, StoreDistValue
    from .Tensor import Tensor
    from .reg import Reg, MaskReg, RegList


Scalar = Union[int, float, Var]
MicroDst = Union["Reg", "MaskReg", "Tensor"]


def _require_micro() -> "MicroModule":
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


def _require_arity(inputs: Tuple[object, ...], expected: int, op: str) -> None:
    if len(inputs) < expected:
        raise TypeError(f"{op}需要至少{expected}个输入，当前数量: {len(inputs)}")


def _as_reg(value: object, arg: str, op: str) -> "Reg":
    from .reg import Reg

    if not isinstance(value, Reg):
        raise TypeError(f"{op}的{arg}必须是Reg类型，当前类型: {type(value)}")
    return value


def _as_reglist(value: object, arg: str, op: str) -> "RegList":
    from .reg import RegList

    if not isinstance(value, RegList):
        raise TypeError(f"{op}的{arg}必须是RegList类型，当前类型: {type(value)}")
    return value


def _as_maskreg(value: object, arg: str, op: str) -> "MaskReg":
    from .reg import MaskReg

    if not isinstance(value, MaskReg):
        raise TypeError(f"{op}的{arg}必须是MaskReg类型，当前类型: {type(value)}")
    return value


def _as_tensor(value: object, arg: str, op: str) -> "Tensor":
    from .Tensor import Tensor

    if not isinstance(value, Tensor):
        raise TypeError(f"{op}的{arg}必须是Tensor类型，当前类型: {type(value)}")
    return value


def _as_scalar(value: object, arg: str, op: str) -> Scalar:
    if not isinstance(value, (Var, int, float)):
        raise TypeError(f"{op}的{arg}必须是Var/int/float，当前类型: {type(value)}")
    return value


def _as_scalar_or_reg(value: object, arg: str, op: str) -> Union[Scalar, "Reg"]:
    from .reg import Reg

    if not isinstance(value, (Reg, Var, int, float)):
        raise TypeError(f"{op}的{arg}必须是Reg/Var/int/float，当前类型: {type(value)}")
    return value


def _as_bool(value: object, arg: str, op: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{op}的{arg}必须是bool类型，当前类型: {type(value)}")
    return value


def _as_compare_mode(value: object, op: str) -> CompareModeType:
    if not isinstance(value, CompareModeType):
        raise TypeError(f"{op}需要CompareModeType，当前类型: {type(value)}")
    return value


def _as_blk_stride(value: object, arg: str, op: str) -> Union[int, Var]:
    if not isinstance(value, (int, Var)):
        raise TypeError(f"{op}的{arg}必须是int或Var类型，当前类型: {type(value)}")
    return value


def _as_loaddist(value: object, op: str) -> "LoadDistValue":
    from ..stub_functions.micro.datamove import LoadDistValue

    if not isinstance(value, LoadDistValue):
        raise TypeError(f"{op}需要LoadDistValue，当前类型: {type(value)}")
    return value


def _as_storedist(value: object, op: str) -> "StoreDistValue":
    from ..stub_functions.micro.datamove import StoreDistValue

    if not isinstance(value, StoreDistValue):
        raise TypeError(f"{op}需要StoreDistValue，当前类型: {type(value)}")
    return value


def _normalize_mask(mask: Optional["MaskReg"], op: str) -> Optional["MaskReg"]:
    from .reg import MaskReg

    if mask is not None and not isinstance(mask, MaskReg):
        raise TypeError(f"{op}的mask必须是MaskReg类型，当前类型: {type(mask)}")
    return mask


class RegOP:
    def __init__(self, opname: str, *inputs: object) -> None:
        self.inputs: Tuple[object, ...] = inputs
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
        from .reg import Reg, RegList

        if self.opname == "cast":
            if len(self.inputs) >= 3 and isinstance(self.inputs[2], DataTypeValue):
                return self.inputs[2]
        if self.inputs and isinstance(self.inputs[0], Reg):
            return self.inputs[0].dtype
        if self.inputs and isinstance(self.inputs[0], RegList):
            if self.opname in ("cadd", "cmax", "cmin"):
                return self.inputs[0].dtype
            raise TypeError(f"无法推断RegList算子{self.opname}的输出dtype")
        if not self.inputs:
            raise TypeError("无法推断RegOP输出dtype")
        raise TypeError("无法推断RegOP输出dtype")

    def _emit_reglist_reduce(
        self,
        dst: "Reg",
        src: "RegList",
        op: str,
        mask: Optional["MaskReg"],
    ) -> None:
        from ..stub_functions import micro

        if src.length <= 0:
            raise ValueError("RegList长度必须大于0")

        if op == "cmax":
            pair_func = micro.vmax
            reduce_func = micro.cmax
        elif op == "cmin":
            pair_func = micro.vmin
            reduce_func = micro.cmin
        elif op == "cadd":
            pair_func = micro.add
            reduce_func = micro.cadd
        else:
            raise ValueError(f"RegList不支持{op}规约")

        if src.length == 1:
            reduce_func(dst, src[0], mask=mask)
            return

        pair_func(dst, src[0], src[1], mask=mask)
        for i in range(2, src.length):
            pair_func(dst, dst, src[i], mask=mask)
        reduce_func(dst, dst, mask=mask)

    def emit(self, dst: MicroDst) -> None:
        from ..stub_functions import micro
        from .reg import RegList

        op = self.opname
        mask = _normalize_mask(self._mask, op)

        if op in ("add", "sub", "mul", "div", "vmax", "vmin", "vand", "vor", "vxor", "prelu"):
            _require_arity(self.inputs, 2, op)
            dst_reg = _as_reg(dst, "dst", op)
            src1 = _as_reg(self.inputs[0], "src1", op)
            src2 = _as_reg(self.inputs[1], "src2", op)
            micro_func = getattr(micro, op)
            micro_func(dst_reg, src1, src2, mask=mask)
            return

        if op in ("exp", "abs", "relu", "sqrt", "ln", "log", "log2", "log10", "neg", "vnot", "vcopy"):
            _require_arity(self.inputs, 1, op)
            dst_reg = _as_reg(dst, "dst", op)
            src = _as_reg(self.inputs[0], "src", op)
            micro_func = getattr(micro, op)
            micro_func(dst_reg, src, mask=mask)
            return

        if op in ("vmaxs", "vmins", "adds", "muls", "lrelu", "shiftls", "shiftrs", "axpy"):
            _require_arity(self.inputs, 2, op)
            dst_reg = _as_reg(dst, "dst", op)
            src = _as_reg(self.inputs[0], "src", op)
            if op in ("shiftls", "shiftrs"):
                value = _as_blk_stride(self.inputs[1], "value", op)
            else:
                value = _as_scalar(self.inputs[1], "value", op)
            micro_func = getattr(micro, op)
            micro_func(dst_reg, src, value, mask=mask)
            return

        if op in ("cmax", "cmin", "cadd"):
            _require_arity(self.inputs, 1, op)
            dst_reg = _as_reg(dst, "dst", op)
            src0 = self.inputs[0]
            if isinstance(src0, RegList):
                self._emit_reglist_reduce(dst_reg, src0, op, mask)
                return
            src = _as_reg(src0, "src", op)
            micro_func = getattr(micro, op)
            micro_func(dst_reg, src, mask=mask)
            return

        if op in ("cgmax", "cgmin", "cgadd", "cpadd"):
            _require_arity(self.inputs, 1, op)
            dst_reg = _as_reg(dst, "dst", op)
            src = _as_reg(self.inputs[0], "src", op)
            micro_func = getattr(micro, op)
            micro_func(dst_reg, src, mask=mask)
            return

        if op == "dup":
            _require_arity(self.inputs, 1, op)
            dst_reg = _as_reg(dst, "dst", op)
            src = _as_scalar_or_reg(self.inputs[0], "src", op)
            micro.dup(dst_reg, src, mask=mask)
            return

        if op == "arange":
            _require_arity(self.inputs, 1, op)
            dst_reg = _as_reg(dst, "dst", op)
            start = _as_scalar(self.inputs[0], "start", op)
            increase = True
            if len(self.inputs) >= 2:
                increase = _as_bool(self.inputs[1], "increase", op)
            micro.arange(dst_reg, start, increase=increase)
            return

        if op == "cast":
            _require_arity(self.inputs, 2, op)
            dst_reg = _as_reg(dst, "dst", op)
            src = _as_reg(self.inputs[0], "src", op)
            cfg = self.inputs[1]
            if cfg is not None and not isinstance(cfg, CastConfig):
                raise TypeError(f"{op}的config必须是CastConfig或None，当前类型: {type(cfg)}")
            micro.cast(dst_reg, src, cfg, mask)
            return

        if op == "compare":
            _require_arity(self.inputs, 3, op)
            dst_mask = _as_maskreg(dst, "dst", op)
            src1 = _as_reg(self.inputs[0], "src1", op)
            src2 = _as_scalar_or_reg(self.inputs[1], "src2", op)
            mode = _as_compare_mode(self.inputs[2], op)
            micro.compare(dst_mask, src1, src2, mode, mask=mask)
            return

        if op == "select":
            _require_arity(self.inputs, 2, op)
            dst_reg = _as_reg(dst, "dst", op)
            src1 = _as_reg(self.inputs[0], "src1", op)
            src2 = _as_reg(self.inputs[1], "src2", op)
            micro.select(dst_reg, src1, src2, mask=mask)
            return

        if op == "mask_not":
            _require_arity(self.inputs, 1, op)
            dst_mask = _as_maskreg(dst, "dst", op)
            src = _as_maskreg(self.inputs[0], "src", op)
            micro.mask_not(dst_mask, src, mask=mask)
            return

        if op in ("mask_and", "mask_or", "mask_xor"):
            _require_arity(self.inputs, 2, op)
            dst_mask = _as_maskreg(dst, "dst", op)
            src1 = _as_maskreg(self.inputs[0], "src1", op)
            src2 = _as_maskreg(self.inputs[1], "src2", op)
            micro_func = getattr(micro, op)
            micro_func(dst_mask, src1, src2, mask=mask)
            return

        if op == "mask_mov":
            _require_arity(self.inputs, 1, op)
            dst_mask = _as_maskreg(dst, "dst", op)
            src = _as_maskreg(self.inputs[0], "src", op)
            micro.mask_mov(dst_mask, src, mask=mask)
            return

        if op == "mask_sel":
            _require_arity(self.inputs, 2, op)
            dst_mask = _as_maskreg(dst, "dst", op)
            src1 = _as_maskreg(self.inputs[0], "src1", op)
            src2 = _as_maskreg(self.inputs[1], "src2", op)
            micro.mask_sel(dst_mask, src1, src2, mask=mask)
            return

        if op in ("mask_pack", "mask_unpack"):
            _require_arity(self.inputs, 1, op)
            dst_mask = _as_maskreg(dst, "dst", op)
            src = _as_maskreg(self.inputs[0], "src", op)
            low_part = True
            if len(self.inputs) >= 2:
                low_part = _as_bool(self.inputs[1], "low_part", op)
            micro_func = getattr(micro, op)
            micro_func(dst_mask, src, low_part=low_part)
            return

        if op in ("mask_interleave", "mask_deinterleave"):
            _require_arity(self.inputs, 3, op)
            dst0 = _as_maskreg(dst, "dst0", op)
            src0 = _as_maskreg(self.inputs[0], "src0", op)
            src1 = _as_maskreg(self.inputs[1], "src1", op)
            dst1 = _as_maskreg(self.inputs[2], "dst1", op)
            micro_func = getattr(micro, op)
            micro_func(dst0, dst1, src0, src1)
            return

        if op == "move_mask_spr":
            dst_mask = _as_maskreg(dst, "dst", op)
            micro.move_mask_spr(dst_mask)
            return

        if op == "update_mask":
            _require_arity(self.inputs, 1, op)
            dst_mask = _as_maskreg(dst, "dst", op)
            cnt = self.inputs[0]
            if not isinstance(cnt, Var):
                raise TypeError(f"{op}的cnt必须是Var类型，当前类型: {type(cnt)}")
            micro.update_mask(dst_mask, cnt)
            return

        if op in ("interleave", "deinterleave"):
            _require_arity(self.inputs, 3, op)
            dst0 = _as_reg(dst, "dst0", op)
            src0 = _as_reg(self.inputs[0], "src0", op)
            src1 = _as_reg(self.inputs[1], "src1", op)
            dst1 = _as_reg(self.inputs[2], "dst1", op)
            micro_func = getattr(micro, op)
            micro_func(dst0, dst1, src0, src1)
            return

        if op == "ub_to_reg":
            _require_arity(self.inputs, 1, op)
            dst_reg = _as_reg(dst, "dst", op)
            src = _as_tensor(self.inputs[0], "src", op)
            blk_stride = 1
            if len(self.inputs) >= 2:
                blk_stride = _as_blk_stride(self.inputs[1], "blk_stride", op)
            micro.ub_to_reg(dst_reg, src, blk_stride=blk_stride, mask=mask)
            return

        if op == "reg_to_ub":
            _require_arity(self.inputs, 1, op)
            dst_tensor = _as_tensor(dst, "dst", op)
            src = _as_reg(self.inputs[0], "src", op)
            blk_stride = 1
            if len(self.inputs) >= 2:
                blk_stride = _as_blk_stride(self.inputs[1], "blk_stride", op)
            micro.reg_to_ub(dst_tensor, src, blk_stride=blk_stride, mask=mask)
            return

        if op == "ub_to_reg_continuous":
            _require_arity(self.inputs, 2, op)
            dst_reg = _as_reg(dst, "dst", op)
            src = _as_tensor(self.inputs[0], "src", op)
            loaddist = _as_loaddist(self.inputs[1], op)
            micro.ub_to_reg_continuous(dst_reg, src, loaddist)
            return

        if op == "reg_to_ub_continuous":
            _require_arity(self.inputs, 2, op)
            dst_tensor = _as_tensor(dst, "dst", op)
            src = _as_reg(self.inputs[0], "src", op)
            storedist = _as_storedist(self.inputs[1], op)
            micro.reg_to_ub_continuous(dst_tensor, src, mask, storedist)
            return

        if op in ("reg_to_ub_downsample", "reg_to_ub_pack4", "reg_to_ub_single"):
            _require_arity(self.inputs, 1, op)
            dst_tensor = _as_tensor(dst, "dst", op)
            src = _as_reg(self.inputs[0], "src", op)
            micro_func = getattr(micro, op)
            micro_func(dst_tensor, src, mask=mask)
            return

        if op in (
            "ub_to_reg_single",
            "ub_to_reg_upsample",
            "ub_to_reg_downsample",
            "ub_to_reg_unpack",
            "ub_to_reg_unpack4",
            "ub_to_reg_brcb",
        ):
            _require_arity(self.inputs, 1, op)
            dst_reg = _as_reg(dst, "dst", op)
            src = _as_tensor(self.inputs[0], "src", op)
            micro_func = getattr(micro, op)
            micro_func(dst_reg, src)
            return

        if op == "ub_to_reg_gather":
            _require_arity(self.inputs, 2, op)
            dst_reg = _as_reg(dst, "dst", op)
            src = _as_tensor(self.inputs[0], "src", op)
            index = _as_reg(self.inputs[1], "index", op)
            micro.ub_to_reg_gather(dst_reg, src, index, mask=mask)
            return

        if op == "reg_to_ub_scatter":
            _require_arity(self.inputs, 2, op)
            dst_tensor = _as_tensor(dst, "dst", op)
            src = _as_reg(self.inputs[0], "src", op)
            index = _as_reg(self.inputs[1], "index", op)
            micro.reg_to_ub_scatter(dst_tensor, src, index, mask=mask)
            return

        if op == "gather":
            _require_arity(self.inputs, 2, op)
            dst_reg = _as_reg(dst, "dst", op)
            src = _as_reg(self.inputs[0], "src", op)
            index = _as_reg(self.inputs[1], "index", op)
            micro.gather(dst_reg, src, index)
            return

        if op == "gather_mask":
            _require_arity(self.inputs, 1, op)
            dst_reg = _as_reg(dst, "dst", op)
            src = _as_reg(self.inputs[0], "src", op)
            micro.gather_mask(dst_reg, src, mask=mask)
            return

        raise ValueError(f"Unsupported RegOP: {op}")

    def run_regop(self) -> "Reg":
        from .reg import Reg, RegList

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

        if self.inputs and isinstance(self.inputs[0], RegList):
            if self.opname not in ("cadd", "cmax", "cmin"):
                raise ValueError("RegList仅支持cadd/cmax/cmin直接生成Reg结果")

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
