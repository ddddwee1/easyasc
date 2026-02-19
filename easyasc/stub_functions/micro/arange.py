from typing import Union

from ...utils.reg import Reg
from ...utils.var import Var
from ...utils.instruction import Instruction
from .microutils import format_scalar, require_micro


def arange(dst: Reg, start: Union[Var, int, float], increase: bool = True) -> None:
    micro = require_micro()
    if not isinstance(dst, Reg):
        raise TypeError(f"dst must be Reg type, current type: {type(dst)}")
    if not isinstance(start, (Var, int, float)):
        raise TypeError(f"start must be Var or numeric value type, current type: {type(start)}")
    if not isinstance(increase, bool):
        raise TypeError(f"increase must be booltype, current type: {type(increase)}") 
    mode = "MicroAPI::IndexOrder::INCREASE_ORDER" if increase else "MicroAPI::IndexOrder::DECREASE_ORDER"
    v = format_scalar(start, dst.dtype)
    micro.instructions.append(
        Instruction("micro_arange", dst=dst, v=v, dtype=dst.dtype, mode=mode)
    )
