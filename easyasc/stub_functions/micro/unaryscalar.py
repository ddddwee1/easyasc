from typing import Optional, Union

from ...utils.reg import Reg, MaskReg
from ...utils.var import Var
from ...utils.instruction import Instruction
from .microutils import ensure_mask, format_scalar, require_micro


def _unary_scalar_op(
    opname: str,
    dst: Reg,
    src: Reg,
    value: Union[Var, int, float],
    mask: Optional[MaskReg],
) -> None:
    micro = require_micro()
    if not isinstance(dst, Reg):
        raise TypeError(f"dst必须是Reg类型，当前类型: {type(dst)}")
    if not isinstance(src, Reg):
        raise TypeError(f"src必须是Reg类型，当前类型: {type(src)}")
    if not isinstance(value, (Var, int, float)):
        raise TypeError(f"value必须是Var或数值类型，当前类型: {type(value)}")
    if dst.dtype != src.dtype:
        raise ValueError("dst/src的数据类型必须一致")

    mask = ensure_mask(mask, dst.dtype, micro)
    v = format_scalar(value, src.dtype)
    micro.instructions.append(Instruction(opname, dst=dst, src=src, v=v, mask=mask))


def vmaxs(dst: Reg, src: Reg, value: Union[Var, int, float], mask: Optional[MaskReg] = None) -> None:
    _unary_scalar_op("micro_vmaxs", dst, src, value, mask)


def vmins(dst: Reg, src: Reg, value: Union[Var, int, float], mask: Optional[MaskReg] = None) -> None:
    _unary_scalar_op("micro_vmins", dst, src, value, mask)


def adds(dst: Reg, src: Reg, value: Union[Var, int, float], mask: Optional[MaskReg] = None) -> None:
    _unary_scalar_op("micro_vadds", dst, src, value, mask)


def muls(dst: Reg, src: Reg, value: Union[Var, int, float], mask: Optional[MaskReg] = None) -> None:
    _unary_scalar_op("micro_vmuls", dst, src, value, mask)


def lrelu(dst: Reg, src: Reg, value: Union[Var, int, float], mask: Optional[MaskReg] = None) -> None:
    _unary_scalar_op("micro_vlrelu", dst, src, value, mask)


def shiftls(dst: Reg, src: Reg, value: Union[Var, int], mask: Optional[MaskReg] = None) -> None:
    if not isinstance(value, (Var, int)):
        raise TypeError(f"value必须是Var或int类型，当前类型: {type(value)}")
    _unary_scalar_op("micro_shiftls", dst, src, value, mask)


def shiftrs(dst: Reg, src: Reg, value: Union[Var, int], mask: Optional[MaskReg] = None) -> None:
    if not isinstance(value, (Var, int)):
        raise TypeError(f"value必须是Var或int类型，当前类型: {type(value)}")
    _unary_scalar_op("micro_shiftrs", dst, src, value, mask)


def axpy(dst: Reg, src: Reg, value: Union[Var, int, float], mask: Optional[MaskReg] = None) -> None:
    _unary_scalar_op("micro_vaxpy", dst, src, value, mask)
