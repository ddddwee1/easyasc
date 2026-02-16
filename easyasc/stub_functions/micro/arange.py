from typing import Union

from ...utils.reg import Reg
from ...utils.var import Var
from ...utils.instruction import Instruction
from .microutils import format_scalar, require_micro


def arange(dst: Reg, start: Union[Var, int, float], increase: bool = True) -> None:
    micro = require_micro()
    if not isinstance(dst, Reg):
        raise TypeError(f"dst必须是Reg类型，当前类型: {type(dst)}")
    if not isinstance(start, (Var, int, float)):
        raise TypeError(f"start必须是Var或数值类型，当前类型: {type(start)}")
    if not isinstance(increase, bool):
        raise TypeError(f"increase必须是bool类型，当前类型: {type(increase)}")

    mode = "MicroAPI::IndexOrder::INCREASE_ORDER" if increase else "MicroAPI::IndexOrder::DECREASE_ORDER"
    v = format_scalar(start, dst.dtype)
    micro.instructions.append(
        Instruction("micro_arange", dst=dst, v=v, dtype=dst.dtype, mode=mode)
    )
