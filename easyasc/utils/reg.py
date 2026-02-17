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


class Reg:
    def __init__(self, dtype: DataTypeValue, name: str = "") -> None:
        if not isinstance(dtype, DataTypeValue):
            raise TypeError(f"dtype必须是DataTypeValue类型，当前类型: {type(dtype)}")
        if not isinstance(name, str):
            raise TypeError(f"name必须是str类型，当前类型: {type(name)}")

        if name == "":
            module = globvars.active_micro
            if module is None:
                raise RuntimeError("active_micro为None，无法自动生成Reg名称")
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
                    raise RuntimeError("active_micro为None，Reg和Tensor赋值仅可在MicroModule中使用")

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
        raise TypeError(f"Reg赋值仅支持RegOP/Reg/Tensor/标量/Var，当前类型: {type(other)}")

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


class MaskReg:
    def __init__(
        self,
        dtype: DataTypeValue,
        init_mode: Optional[MaskTypeValue] = None,
        name: str = "",
    ) -> None:
        if not isinstance(dtype, DataTypeValue):
            raise TypeError(f"dtype必须是DataTypeValue类型，当前类型: {type(dtype)}")
        if init_mode is None:
            init_mode = MaskType.ALL
        if not isinstance(init_mode, MaskTypeValue):
            raise TypeError(f"init_mode必须是MaskTypeValue类型，当前类型: {type(init_mode)}")
        if not isinstance(name, str):
            raise TypeError(f"name必须是str类型，当前类型: {type(name)}")

        if name == "":
            module = globvars.active_micro
            if module is None:
                raise RuntimeError("active_micro为None，无法自动生成MaskReg名称")
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
            raise TypeError(f"mask只能与RegOP相乘，当前类型: {type(other)}")
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
        raise TypeError(f"MaskReg赋值仅支持RegOP/Var，当前类型: {type(other)}")

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
