from typing import Optional

from ...utils.reg import Reg, MaskReg
from ...utils.instruction import Instruction
from .microutils import ensure_mask, require_micro


def _unary_op(opname: str, dst: Reg, src: Reg, mask: Optional[MaskReg]) -> None:
    micro = require_micro()
    if not isinstance(dst, Reg):
        raise TypeError(f"dst必须是Reg类型，当前类型: {type(dst)}")
    if not isinstance(src, Reg):
        raise TypeError(f"src必须是Reg类型，当前类型: {type(src)}")
    if dst.dtype != src.dtype:
        raise ValueError("dst/src的数据类型必须一致")

    mask = ensure_mask(mask, dst.dtype, micro)
    micro.instructions.append(Instruction(opname, dst=dst, src=src, mask=mask))


def exp(dst: Reg, src: Reg, mask: Optional[MaskReg] = None) -> None:
    _unary_op("micro_vexp", dst, src, mask)


def abs(dst: Reg, src: Reg, mask: Optional[MaskReg] = None) -> None:
    _unary_op("micro_vabs", dst, src, mask)


def relu(dst: Reg, src: Reg, mask: Optional[MaskReg] = None) -> None:
    _unary_op("micro_vrelu", dst, src, mask)


def sqrt(dst: Reg, src: Reg, mask: Optional[MaskReg] = None) -> None:
    _unary_op("micro_vsqrt", dst, src, mask)


def ln(dst: Reg, src: Reg, mask: Optional[MaskReg] = None) -> None:
    _unary_op("micro_vln", dst, src, mask)


def log(dst: Reg, src: Reg, mask: Optional[MaskReg] = None) -> None:
    _unary_op("micro_vlog", dst, src, mask)


def log2(dst: Reg, src: Reg, mask: Optional[MaskReg] = None) -> None:
    _unary_op("micro_vlog2", dst, src, mask)


def log10(dst: Reg, src: Reg, mask: Optional[MaskReg] = None) -> None:
    _unary_op("micro_vlog10", dst, src, mask)


def neg(dst: Reg, src: Reg, mask: Optional[MaskReg] = None) -> None:
    _unary_op("micro_vneg", dst, src, mask)


def vnot(dst: Reg, src: Reg, mask: Optional[MaskReg] = None) -> None:
    _unary_op("micro_vnot", dst, src, mask)


def vcopy(dst: Reg, src: Reg, mask: Optional[MaskReg] = None) -> None:
    _unary_op("micro_vcopy", dst, src, mask)
