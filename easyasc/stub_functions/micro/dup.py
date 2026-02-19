from typing import Optional, Union

from ...utils.reg import Reg, MaskReg
from ...utils.var import Var
from ...utils.instruction import Instruction
from .microutils import ensure_mask, format_scalar, require_micro


def dup(dst: Reg, src: Union[Reg, float, int, Var], mask: Optional[MaskReg] = None) -> None:
    micro = require_micro()
    if not isinstance(dst, Reg):
        raise TypeError(f"dst must be Reg type, current type: {type(dst)}")
    if not isinstance(src, (Reg, Var, int, float)):
        raise TypeError(f"src must be Reg or numeric value/Var type, current type: {type(src)}")

    mask = ensure_mask(mask, dst.dtype, micro)

    if isinstance(src, (Reg, Var)):
        v = src
    else:
        v = format_scalar(src, dst.dtype)

    micro.instructions.append(Instruction("micro_vdup", dst=dst, src=v, mask=mask))
