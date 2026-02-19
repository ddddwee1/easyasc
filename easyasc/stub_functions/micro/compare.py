from typing import Optional, Union

from ...utils.comparemode import CompareModeType
from ...utils.reg import Reg, MaskReg
from ...utils.var import Var
from ...utils.instruction import Instruction
from .microutils import ensure_mask, format_scalar, require_micro


def compare(
    dst: MaskReg,
    src1: Reg,
    src2: Union[Var, int, float, Reg],
    mode: CompareModeType,
    mask: Optional[MaskReg] = None,
) -> None:
    micro = require_micro()
    if not isinstance(dst, MaskReg):
        raise TypeError(f"dst must be MaskReg type, current type: {type(dst)}")
    if not isinstance(src1, Reg):
        raise TypeError(f"src1 must be Reg type, current type: {type(src1)}")
    if not isinstance(src2, (Var, int, float, Reg)):
        raise TypeError(f"src2 must be Reg or numeric value/Var type, current type: {type(src2)}")
    if not isinstance(mode, CompareModeType):
        raise TypeError(f"mode must be CompareModeType type, current type: {type(mode)}")
    if dst.dtype != src1.dtype:
        raise ValueError("dst/src1 data types must match") 
    mask = ensure_mask(mask, dst.dtype, micro)

    if isinstance(src2, Reg):
        if src1.dtype != src2.dtype:
            raise ValueError("src1/src2 data types must match")
        micro.instructions.append(
            Instruction(
                "micro_compare",
                dst=dst,
                src1=src1,
                src2=src2,
                dtype=dst.dtype,
                mode=mode,
                mask=mask,
            )
        )
    else:
        v = format_scalar(src2, src1.dtype)
        micro.instructions.append(
            Instruction(
                "micro_compares",
                dst=dst,
                src1=src1,
                src2=v,
                dtype=dst.dtype,
                mode=mode,
                mask=mask,
            )
        )


def select(dst: Reg, src1: Reg, src2: Reg, mask: Optional[MaskReg] = None) -> None:
    micro = require_micro()
    if not isinstance(dst, Reg):
        raise TypeError(f"dst must be Reg type, current type: {type(dst)}")
    if not isinstance(src1, Reg):
        raise TypeError(f"src1 must be Reg type, current type: {type(src1)}")
    if not isinstance(src2, Reg):
        raise TypeError(f"src2 must be Reg type, current type: {type(src2)}")
    if src1.dtype != src2.dtype or dst.dtype != src1.dtype:
        raise ValueError("dst/src1/src2 data types must match") 
    mask = ensure_mask(mask, dst.dtype, micro)
    micro.instructions.append(
        Instruction("micro_select", dst=dst, src1=src1, src2=src2, mask=mask)
    )
