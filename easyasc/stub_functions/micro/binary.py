from typing import Optional

from ...utils.reg import Reg, MaskReg
from ...utils.instruction import Instruction
from .microutils import ensure_mask, require_micro


def _binary_op(opname: str, dst: Reg, src1: Reg, src2: Reg, mask: Optional[MaskReg]) -> None:
    micro = require_micro()
    if not isinstance(dst, Reg):
        raise TypeError(f"dst必须是Reg类型，当前类型: {type(dst)}")
    if not isinstance(src1, Reg):
        raise TypeError(f"src1必须是Reg类型，当前类型: {type(src1)}")
    if not isinstance(src2, Reg):
        raise TypeError(f"src2必须是Reg类型，当前类型: {type(src2)}")
    if dst.dtype != src1.dtype or dst.dtype != src2.dtype:
        raise ValueError("dst/src1/src2的数据类型必须一致")

    mask = ensure_mask(mask, dst.dtype, micro)
    micro.instructions.append(
        Instruction(opname, dst=dst, src1=src1, src2=src2, mask=mask)
    )


def vmax(dst: Reg, src1: Reg, src2: Reg, mask: Optional[MaskReg] = None) -> None:
    _binary_op("micro_vmax", dst, src1, src2, mask)


def vmin(dst: Reg, src1: Reg, src2: Reg, mask: Optional[MaskReg] = None) -> None:
    _binary_op("micro_vmin", dst, src1, src2, mask)


def add(dst: Reg, src1: Reg, src2: Reg, mask: Optional[MaskReg] = None) -> None:
    _binary_op("micro_vadd", dst, src1, src2, mask)


def sub(dst: Reg, src1: Reg, src2: Reg, mask: Optional[MaskReg] = None) -> None:
    _binary_op("micro_vsub", dst, src1, src2, mask)


def mul(dst: Reg, src1: Reg, src2: Reg, mask: Optional[MaskReg] = None) -> None:
    _binary_op("micro_vmul", dst, src1, src2, mask)


def div(dst: Reg, src1: Reg, src2: Reg, mask: Optional[MaskReg] = None) -> None:
    _binary_op("micro_vdiv", dst, src1, src2, mask)


def vand(dst: Reg, src1: Reg, src2: Reg, mask: Optional[MaskReg] = None) -> None:
    _binary_op("micro_vand", dst, src1, src2, mask)


def vor(dst: Reg, src1: Reg, src2: Reg, mask: Optional[MaskReg] = None) -> None:
    _binary_op("micro_vor", dst, src1, src2, mask)


def vxor(dst: Reg, src1: Reg, src2: Reg, mask: Optional[MaskReg] = None) -> None:
    _binary_op("micro_vxor", dst, src1, src2, mask)


def prelu(dst: Reg, src1: Reg, src2: Reg, mask: Optional[MaskReg] = None) -> None:
    _binary_op("micro_vprelu", dst, src1, src2, mask)
