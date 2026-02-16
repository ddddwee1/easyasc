from ...utils.reg import Reg
from ...utils.instruction import Instruction
from .microutils import require_micro


def deinterleave(dst0: Reg, dst1: Reg, src0: Reg, src1: Reg) -> None:
    micro = require_micro()
    for name, reg in ("dst0", dst0), ("dst1", dst1), ("src0", src0), ("src1", src1):
        if not isinstance(reg, Reg):
            raise TypeError(f"{name}必须是Reg类型，当前类型: {type(reg)}")
    if dst0.dtype != dst1.dtype or src0.dtype != src1.dtype or dst0.dtype != src0.dtype:
        raise ValueError("dst/src的数据类型必须一致")

    micro.instructions.append(
        Instruction("micro_dinterleave", dst0=dst0, dst1=dst1, src0=src0, src1=src1)
    )


def interleave(dst0: Reg, dst1: Reg, src0: Reg, src1: Reg) -> None:
    micro = require_micro()
    for name, reg in ("dst0", dst0), ("dst1", dst1), ("src0", src0), ("src1", src1):
        if not isinstance(reg, Reg):
            raise TypeError(f"{name}必须是Reg类型，当前类型: {type(reg)}")
    if dst0.dtype != dst1.dtype or src0.dtype != src1.dtype or dst0.dtype != src0.dtype:
        raise ValueError("dst/src的数据类型必须一致")

    micro.instructions.append(
        Instruction("micro_interleave", dst0=dst0, dst1=dst1, src0=src0, src1=src1)
    )
