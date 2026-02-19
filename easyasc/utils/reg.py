from typing import Optional, TYPE_CHECKING, Union

from .. import globvars
from .castconfig import CastConfig
from .comparemode import CompareMode
from .datatype import DataTypeValue
from .instruction import Instruction
from .mask import MaskType, MaskTypeValue
from .var import Var

if TYPE_CHECKING:
    from .Tensor import Tensor
    from .regop import RegOP


Scalar = Union[int, float, Var]


def _make_reg_proxy(dtype: DataTypeValue, name: str) -> "Reg":
    reg = object.__new__(Reg)
    reg.dtype = dtype
    reg.name = name
    return reg


class Reg:
    def __init__(self, dtype: DataTypeValue, name: str = "") -> None:
        if not isinstance(dtype, DataTypeValue):
            raise TypeError(f"dtype must be DataTypeValue, got: {type(dtype)}")
        if not isinstance(name, str):
            raise TypeError(f"name must be str, got: {type(name)}")

        if name == "":
            module = globvars.active_micro
            if module is None:
                raise RuntimeError("active_micro is None, cannot auto-generate Reg name")
            idx = module.tmp_idx
            module.tmp_idx += 1
            name = f"_reg_{idx}"

        self.dtype = dtype
        self.name = name
        if globvars.active_micro is not None:
            globvars.active_micro.instructions.append(Instruction("create_reg", reg=self))

    def __repr__(self) -> str:
        return f"Reg(name={self.name!r}, dtype={self.dtype!r})"

    def __str__(self) -> str:
        return self.name

    def __add__(self, other):
        from .regop import RegOP
        if isinstance(other, RegOP):
            other = other.run_regop()
        if isinstance(other, Reg):
            return RegOP("add", self, other)
        if isinstance(other, (int, float, Var)):
            return RegOP("adds", self, other)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        from .regop import RegOP
        if isinstance(other, RegOP):
            other = other.run_regop()
        if isinstance(other, Reg):
            return RegOP("sub", self, other)
        if isinstance(other, (int, float, Var)):
            return RegOP("adds", self, -1 * other)
        return NotImplemented

    def __rsub__(self, other):
        from .regop import RegOP
        if isinstance(other, RegOP):
            other = other.run_regop()
            return RegOP("sub", other, self)
        if isinstance(other, Reg):
            return RegOP("sub", other, self)
        return NotImplemented

    def __mul__(self, other):
        from .regop import RegOP
        if isinstance(other, RegOP):
            other = other.run_regop()
        if isinstance(other, Reg):
            return RegOP("mul", self, other)
        if isinstance(other, (int, float, Var)):
            return RegOP("muls", self, other)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        from .regop import RegOP
        if isinstance(other, RegOP):
            other = other.run_regop()
        if isinstance(other, Reg):
            return RegOP("div", self, other)
        if isinstance(other, (int, float, Var)):
            return RegOP("muls", self, 1 / other)
        return NotImplemented

    def __rtruediv__(self, other):
        from .regop import RegOP
        if isinstance(other, RegOP):
            other = other.run_regop()
            return RegOP("div", other, self)
        if isinstance(other, Reg):
            return RegOP("div", other, self)
        return NotImplemented

    def __ge__(self, other):
        from .regop import RegOP
        return RegOP("compare", self, other, CompareMode.GE)

    def __gt__(self, other):
        from .regop import RegOP
        return RegOP("compare", self, other, CompareMode.GT)

    def __le__(self, other):
        from .regop import RegOP
        return RegOP("compare", self, other, CompareMode.LE)

    def __lt__(self, other):
        from .regop import RegOP
        return RegOP("compare", self, other, CompareMode.LT)

    def __eq__(self, other):
        from .regop import RegOP
        return RegOP("compare", self, other, CompareMode.EQ)

    def __ne__(self, other):
        from .regop import RegOP
        return RegOP("compare", self, other, CompareMode.NE)

    def __ilshift__(self, other):
        from .Tensor import Tensor
        from .regop import RegOP
        from ..stub_functions.micro.datamove import (
            ub_to_reg,
            ub_to_reg_brcb,
            ub_to_reg_upsample,
            ub_to_reg_unpack,
            ub_to_reg_unpack4,
            ub_to_reg_downsample,
            ub_to_reg_single,
        )

        if isinstance(other, RegOP):
            other.release_inputs()
            other.emit(self)
            return self
        if isinstance(other, Tensor):
            mode = getattr(other, "vec_copy_mode", "")
            if other.dtype!=self.dtype:
                from ..stub_functions.micro.cast import cast

                micro = globvars.active_micro
                if micro is None:
                    raise RuntimeError("active_micro is None, Reg/Tensor assignment is only allowed in MicroModule")

                tmp_reg = micro.get_reg(other.dtype)

                dst_size = self.dtype.size
                src_size = other.dtype.size
                if mode == "single":
                    ub_to_reg_single(tmp_reg, other)
                elif dst_size == 4 and src_size == 2:
                    ub_to_reg_unpack(tmp_reg, other)
                elif dst_size == 2 and src_size == 1:
                    ub_to_reg_unpack(tmp_reg, other)
                elif dst_size == 4 and src_size == 1:
                    ub_to_reg_unpack4(tmp_reg, other)
                elif dst_size < src_size:
                    ub_to_reg(tmp_reg, other)
                else:
                    raise TypeError(f"Dataload+Cast does not support from {other.dtype} to {self.dtype}")
                cast(self, tmp_reg)
                micro.release_reg(tmp_reg)
            else:
                if mode == "brcb":
                    ub_to_reg_brcb(self, other)
                elif mode == "upsample":
                    ub_to_reg_upsample(self, other)
                elif mode == "unpack":
                    ub_to_reg_unpack(self, other)
                elif mode == "unpack4":
                    ub_to_reg_unpack4(self, other)
                elif mode == "downsample":
                    ub_to_reg_downsample(self, other)
                elif mode == "single":
                    ub_to_reg_single(self, other)
                else:
                    ub_to_reg(self, other)
            return self
        if isinstance(other, Reg):
            from ..stub_functions.micro.unary import vcopy
            vcopy(self, other)
            return self
        if isinstance(other, (int, float, Var)):
            from ..stub_functions.micro.dup import dup
            dup(self, other)
            return self
        raise TypeError(f"Reg assignment only supports RegOP/Reg/Tensor/scalar/Var, got: {type(other)}")

    def cast(self, cfg: Optional[CastConfig] = None):
        from .regop import RegOP
        if cfg is None:
            cfg = CastConfig()
        return RegOP("cast", self, cfg)

    def astype(self, dtype: DataTypeValue, cfg: Optional[CastConfig] = None):
        from .regop import RegOP
        return RegOP("cast", self, cfg, dtype)

    def exp(self):
        from .regop import RegOP
        return RegOP("exp", self)

    def abs(self):
        from .regop import RegOP
        return RegOP("abs", self)

    def sqrt(self):
        from .regop import RegOP
        return RegOP("sqrt", self)

    def relu(self):
        from .regop import RegOP
        return RegOP("relu", self)

    def ln(self):
        from .regop import RegOP
        return RegOP("ln", self)

    def log(self):
        from .regop import RegOP
        return RegOP("log", self)

    def log2(self):
        from .regop import RegOP
        return RegOP("log2", self)

    def log10(self):
        from .regop import RegOP
        return RegOP("log10", self)

    def neg(self):
        from .regop import RegOP
        return RegOP("neg", self)

    def vnot(self):
        from .regop import RegOP
        return RegOP("vnot", self)

    def vcopy(self):
        from .regop import RegOP
        return RegOP("vcopy", self)

    def shiftls(self, value: Scalar):
        from .regop import RegOP
        return RegOP("shiftls", self, value)

    def shiftrs(self, value: Scalar):
        from .regop import RegOP
        return RegOP("shiftrs", self, value)

    def axpy(self, value: Scalar):
        from .regop import RegOP
        return RegOP("axpy", self, value)

    def lrelu(self, value: Scalar):
        from .regop import RegOP
        return RegOP("lrelu", self, value)
    
    def vmins(self, value: Scalar):
        from .regop import RegOP
        return RegOP("vmins", self, value)
    
    def vmaxs(self, value: Scalar):
        from .regop import RegOP
        return RegOP("vmaxs", self, value)

    def vand(self, other: "Reg"):
        from .regop import RegOP
        return RegOP("vand", self, other)

    def vor(self, other: "Reg"):
        from .regop import RegOP
        return RegOP("vor", self, other)

    def vxor(self, other: "Reg"):
        from .regop import RegOP
        return RegOP("vxor", self, other)

    def prelu(self, other: "Reg"):
        from .regop import RegOP
        return RegOP("prelu", self, other)

    def vmax(self, other: "Reg"):
        from .regop import RegOP
        return RegOP("vmax", self, other)

    def vmin(self, other: "Reg"):
        from .regop import RegOP
        return RegOP("vmin", self, other)

    def cadd(self):
        from .regop import RegOP
        return RegOP("cadd", self)

    def cmax(self):
        from .regop import RegOP
        return RegOP("cmax", self)

    def cmin(self):
        from .regop import RegOP
        return RegOP("cmin", self)

    def cgadd(self):
        from .regop import RegOP
        return RegOP("cgadd", self)

    def cgmax(self):
        from .regop import RegOP
        return RegOP("cgmax", self)

    def cgmin(self):
        from .regop import RegOP
        return RegOP("cgmin", self)

    def cpadd(self):
        from .regop import RegOP
        return RegOP("cpadd", self)

    def dup(self):
        from .regop import RegOP
        return RegOP("dup", self)

    def fill(self, value: Scalar) -> None:
        from ..stub_functions.micro.dup import dup
        dup(self, value)

    def arange(self, start: Scalar, increase: bool = True) -> None:
        from ..stub_functions.micro.arange import arange
        arange(self, start, increase=increase)

    def downsample(self):
        from .regop import RegOP
        return RegOP("reg_to_ub_downsample", self)

    def pack4(self):
        from .regop import RegOP
        return RegOP("reg_to_ub_pack4", self)

    def single_value(self):
        from .regop import RegOP
        return RegOP("reg_to_ub_single", self)

    def ub_gather(self, src: "Tensor", index: "Reg", mask: Optional["MaskReg"] = None) -> None:
        from ..stub_functions.micro.datamove import ub_to_reg_gather
        ub_to_reg_gather(self, src, index, mask=mask)

    def gather(self, src: "Reg", index: "Reg") -> None:
        from ..stub_functions.micro.datamove import gather
        gather(self, src, index)

    def gather_mask(self, src: "Reg", mask: Optional["MaskReg"] = None) -> None:
        from ..stub_functions.micro.datamove import gather_mask
        gather_mask(self, src, mask=mask)


class RegList:
    def __init__(self, dtype: DataTypeValue, length: int, name: str = "") -> None:
        if not isinstance(dtype, DataTypeValue):
            raise TypeError(f"dtype must be DataTypeValue, got: {type(dtype)}")
        if not isinstance(length, int):
            raise TypeError(f"length must be int, got: {type(length)}")
        if length <= 0:
            raise ValueError(f"length must be greater than 0, current value: {length}")
        if not isinstance(name, str):
            raise TypeError(f"name must be str, got: {type(name)}")

        if name == "":
            module = globvars.active_micro
            if module is None:
                raise RuntimeError("active_micro is None, cannot auto-generate RegList name")
            idx = module.tmp_idx
            module.tmp_idx += 1
            name = f"_reglist_{idx}"

        self.dtype = dtype
        self.length = length
        self.name = name

        if globvars.active_micro is not None:
            globvars.active_micro.instructions.append(
                Instruction("create_reglist", reglist=self)
            )

    def __repr__(self) -> str:
        return (
            f"RegList(name={self.name!r}, dtype={self.dtype!r}, "
            f"length={self.length!r})"
        )

    def __str__(self) -> str:
        return self.name

    def __getitem__(self, idx):
        if not isinstance(idx, (Var, int)):
            raise TypeError(f"RegList index only supports Var/int, got: {type(idx)}")
        reg_name = f"{self.name}[{idx}]"
        return _make_reg_proxy(self.dtype, reg_name)

    def __setitem__(self, idx, value) -> None:
        # `self[i] <<= ...` triggers a follow-up `__setitem__` in Python's
        # augmented-assignment protocol. RegList elements are proxy regs, so
        # there is nothing to store back into the container.
        _ = idx
        _ = value

    def _copy_mask(self, src: "RegOP", dst: "RegOP") -> None:
        if src.mask is not None:
            dst.setmask(src.mask)

    def _emit_unary(self, op: "RegOP") -> None:
        from .regop import RegOP

        if len(op.inputs) < 1:
            raise TypeError(f"{op.opname} requires at least 1 input")
        src = op.inputs[0]
        if not isinstance(src, RegList):
            raise TypeError(f"{op.opname} input must be RegList, got: {type(src)}")
        if src.length != self.length:
            raise ValueError(f"RegList length mismatch: {self.length} vs {src.length}")

        for i in range(self.length):
            item_op = RegOP(op.opname, src[i])
            self._copy_mask(op, item_op)
            self[i] <<= item_op

    def _emit_unary_scalar(self, op: "RegOP") -> None:
        from .regop import RegOP

        if len(op.inputs) < 2:
            raise TypeError(f"{op.opname} requires at least 2 inputs")
        src = op.inputs[0]
        value = op.inputs[1]
        if not isinstance(src, RegList):
            raise TypeError(f"{op.opname} input must be RegList, got: {type(src)}")
        if src.length != self.length:
            raise ValueError(f"RegList length mismatch: {self.length} vs {src.length}")
        if not isinstance(value, (Var, int, float)):
            raise TypeError(f"{op.opname} value must be Var/int/float, got: {type(value)}")

        for i in range(self.length):
            item_op = RegOP(op.opname, src[i], value)
            self._copy_mask(op, item_op)
            self[i] <<= item_op

    def _emit_binary(self, op: "RegOP") -> None:
        from .regop import RegOP

        if len(op.inputs) < 2:
            raise TypeError(f"{op.opname} requires at least 2 inputs")
        src1 = op.inputs[0]
        src2 = op.inputs[1]
        if not isinstance(src1, RegList):
            raise TypeError(f"{op.opname} src1 must be RegList, got: {type(src1)}")
        if src1.length != self.length:
            raise ValueError(f"RegList length mismatch: {self.length} vs {src1.length}")

        if isinstance(src2, RegList):
            if src2.length != self.length:
                raise ValueError(f"RegList length mismatch: {self.length} vs {src2.length}")
            for i in range(self.length):
                item_op = RegOP(op.opname, src1[i], src2[i])
                self._copy_mask(op, item_op)
                self[i] <<= item_op
            return

        if isinstance(src2, Reg):
            for i in range(self.length):
                item_op = RegOP(op.opname, src1[i], src2)
                self._copy_mask(op, item_op)
                self[i] <<= item_op
            return

        if isinstance(src2, (Var, int, float)):
            if op.opname not in ("add", "sub", "mul", "div"):
                raise TypeError(f"{op.opname} does not support RegList with scalar/Var operands")
            for i in range(self.length):
                if op.opname == "add":
                    item_op = src1[i] + src2
                elif op.opname == "sub":
                    item_op = src1[i] - src2
                elif op.opname == "mul":
                    item_op = src1[i] * src2
                else:
                    item_op = src1[i] / src2
                self._copy_mask(op, item_op)
                self[i] <<= item_op
            return

        raise TypeError(f"{op.opname} src2 type is not supported: {type(src2)}")

    def __ilshift__(self, other):
        from .Tensor import Tensor
        from .regop import RegOP

        if isinstance(other, Tensor):
            block = 256 // self.dtype.size
            for i in range(self.length):
                self[i] <<= other[block * i]
            return self

        if isinstance(other, RegOP):
            binary_ops = ("add", "sub", "mul", "div", "vmax", "vmin", "vand", "vor", "vxor", "prelu")
            unary_ops = ("exp", "abs", "sqrt", "relu", "ln", "log", "log2", "log10", "neg", "vnot", "vcopy")
            unary_scalar_ops = ("shiftls", "shiftrs", "axpy", "lrelu", "vmaxs", "vmins", "adds", "muls")
            if other.opname in binary_ops:
                self._emit_binary(other)
                return self
            if other.opname in unary_ops:
                self._emit_unary(other)
                return self
            if other.opname in unary_scalar_ops:
                self._emit_unary_scalar(other)
                return self
            raise ValueError(f"RegList does not support assignment from RegOP: {other.opname}")

        raise TypeError(f"RegList assignment only supports Tensor/RegOP, got: {type(other)}")

    def cmax(self):
        from .regop import RegOP
        return RegOP("cmax", self)

    def cmin(self):
        from .regop import RegOP
        return RegOP("cmin", self)

    def cadd(self):
        from .regop import RegOP
        return RegOP("cadd", self)

    def __add__(self, other):
        from .regop import RegOP
        return RegOP("add", self, other)

    def __sub__(self, other):
        from .regop import RegOP
        return RegOP("sub", self, other)

    def __mul__(self, other):
        from .regop import RegOP
        return RegOP("mul", self, other)

    def __truediv__(self, other):
        from .regop import RegOP
        return RegOP("div", self, other)

    def exp(self):
        from .regop import RegOP
        return RegOP("exp", self)

    def abs(self):
        from .regop import RegOP
        return RegOP("abs", self)

    def sqrt(self):
        from .regop import RegOP
        return RegOP("sqrt", self)

    def relu(self):
        from .regop import RegOP
        return RegOP("relu", self)

    def ln(self):
        from .regop import RegOP
        return RegOP("ln", self)

    def log(self):
        from .regop import RegOP
        return RegOP("log", self)

    def log2(self):
        from .regop import RegOP
        return RegOP("log2", self)

    def log10(self):
        from .regop import RegOP
        return RegOP("log10", self)

    def neg(self):
        from .regop import RegOP
        return RegOP("neg", self)

    def vnot(self):
        from .regop import RegOP
        return RegOP("vnot", self)

    def vcopy(self):
        from .regop import RegOP
        return RegOP("vcopy", self)

    def shiftls(self, value: Union[Var, int]):
        from .regop import RegOP
        return RegOP("shiftls", self, value)

    def shiftrs(self, value: Union[Var, int]):
        from .regop import RegOP
        return RegOP("shiftrs", self, value)

    def axpy(self, value: Scalar):
        from .regop import RegOP
        return RegOP("axpy", self, value)

    def lrelu(self, value: Scalar):
        from .regop import RegOP
        return RegOP("lrelu", self, value)

    def vmaxs(self, value: Scalar):
        from .regop import RegOP
        return RegOP("vmaxs", self, value)

    def vmins(self, value: Scalar):
        from .regop import RegOP
        return RegOP("vmins", self, value)

    def vand(self, other):
        from .regop import RegOP
        return RegOP("vand", self, other)

    def vor(self, other):
        from .regop import RegOP
        return RegOP("vor", self, other)

    def vxor(self, other):
        from .regop import RegOP
        return RegOP("vxor", self, other)

    def prelu(self, other):
        from .regop import RegOP
        return RegOP("prelu", self, other)

    def vmax(self, other):
        from .regop import RegOP
        return RegOP("vmax", self, other)

    def vmin(self, other):
        from .regop import RegOP
        return RegOP("vmin", self, other)

    def fill(self, value: Scalar) -> None:
        for i in range(self.length):
            self[i] <<= value


class MaskReg:
    def __init__(
        self,
        dtype: DataTypeValue,
        init_mode: Optional[MaskTypeValue] = None,
        name: str = "",
    ) -> None:
        if not isinstance(dtype, DataTypeValue):
            raise TypeError(f"dtype must be DataTypeValue, got: {type(dtype)}")
        if init_mode is None:
            init_mode = MaskType.ALL
        if not isinstance(init_mode, MaskTypeValue):
            raise TypeError(f"init_mode must be MaskTypeValue, got: {type(init_mode)}")
        if not isinstance(name, str):
            raise TypeError(f"name must be str, got: {type(name)}")

        if name == "":
            module = globvars.active_micro
            if module is None:
                raise RuntimeError("active_micro is None, cannot auto-generate MaskReg name")
            idx = module.tmp_idx
            module.tmp_idx += 1
            name = f"_maskreg_{idx}"

        self.dtype = dtype
        self.init_mode = init_mode
        self.name = name
        if globvars.active_micro is not None:
            globvars.active_micro.instructions.append(Instruction("create_maskreg", reg=self))

    def __repr__(self) -> str:
        return (
            f"MaskReg(name={self.name!r}, dtype={self.dtype!r}, "
            f"init_mode={self.init_mode!r})"
        )

    def __str__(self) -> str:
        return self.name

    def __mul__(self, other: "RegOP"):
        from .regop import RegOP
        if not isinstance(other, RegOP):
            raise TypeError(f"mask can only multiply with RegOP, got: {type(other)}")
        other.setmask(self)
        return other

    def __rmul__(self, other: "RegOP"):
        return self.__mul__(other)

    def __ilshift__(self, other):
        from .regop import RegOP
        if isinstance(other, Var):
            from ..stub_functions.micro.mask import update_mask
            update_mask(self, other)
            return self
        if isinstance(other, RegOP):
            other.release_inputs()
            other.emit(self)
            return self
        raise TypeError(f"MaskReg assignment only supports RegOP/Var, got: {type(other)}")

    def __invert__(self):
        from .regop import RegOP
        return RegOP("mask_not", self)

    def __and__(self, other: "MaskReg"):
        from .regop import RegOP
        return RegOP("mask_and", self, other)

    def __or__(self, other: "MaskReg"):
        from .regop import RegOP
        return RegOP("mask_or", self, other)

    def __xor__(self, other: "MaskReg"):
        from .regop import RegOP
        return RegOP("mask_xor", self, other)

    def mov(self, src: "MaskReg"):
        from .regop import RegOP
        return RegOP("mask_mov", src)

    def sel(self, src1: "MaskReg", src2: "MaskReg"):
        from .regop import RegOP
        return RegOP("mask_sel", src1, src2)

    def pack(self, low_part: bool = True):
        from .regop import RegOP
        return RegOP("mask_pack", self, low_part)

    def unpack(self, low_part: bool = True):
        from .regop import RegOP
        return RegOP("mask_unpack", self, low_part)

    def move_to_spr(self):
        from .regop import RegOP
        return RegOP("move_mask_spr")

    def update(self, cnt: Var) -> None:
        from ..stub_functions.micro.mask import update_mask
        update_mask(self, cnt)

    def select(self, src1: Reg, src2: Reg):
        from .regop import RegOP
        op = RegOP("select", src1, src2)
        op.setmask(self)
        return op
