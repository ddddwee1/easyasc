from typing import Optional

from ...utils.reg import Reg, MaskReg
from ...utils.instruction import Instruction
from .microutils import ensure_mask, require_micro


def _group_op(opname: str, dst: Reg, src: Reg, mask: Optional[MaskReg]) -> None:
    micro = require_micro()
    if not isinstance(dst, Reg):
        raise TypeError(f"dst must be Reg type, current type: {type(dst)}")
    if not isinstance(src, Reg):
        raise TypeError(f"src must be Reg type, current type: {type(src)}")
    if dst.dtype != src.dtype:
        raise ValueError("dst/src data types must match") 
    mask = ensure_mask(mask, dst.dtype, micro)
    micro.instructions.append(Instruction(opname, dst=dst, src=src, mask=mask))


def cmax(dst: Reg, src: Reg, mask: Optional[MaskReg] = None) -> None:
    _group_op("micro_vcmax", dst, src, mask)


def cgmax(dst: Reg, src: Reg, mask: Optional[MaskReg] = None) -> None:
    _group_op("micro_vcgmax", dst, src, mask)


def cmin(dst: Reg, src: Reg, mask: Optional[MaskReg] = None) -> None:
    _group_op("micro_vcmin", dst, src, mask)


def cgmin(dst: Reg, src: Reg, mask: Optional[MaskReg] = None) -> None:
    _group_op("micro_vcgmin", dst, src, mask)


def cadd(dst: Reg, src: Reg, mask: Optional[MaskReg] = None) -> None:
    _group_op("micro_vcadd", dst, src, mask)


def cgadd(dst: Reg, src: Reg, mask: Optional[MaskReg] = None) -> None:
    _group_op("micro_vcgadd", dst, src, mask)


def cpadd(dst: Reg, src: Reg, mask: Optional[MaskReg] = None) -> None:
    _group_op("micro_vcpadd", dst, src, mask)
