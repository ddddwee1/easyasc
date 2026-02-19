from typing import Union

from ...utils.var import Var
from ...utils.datatype import Datatype
from ...utils.instruction import Instruction
from ... import globvars


def set_mask(mask_high: Union[int, Var], mask_low: Union[int, Var]) -> None:
    if isinstance(mask_high, Var) and mask_high.dtype is not Datatype.uint64:
        raise TypeError(f"mask_high must be uint64type, current type: {mask_high.dtype}")
    if isinstance(mask_low, Var) and mask_low.dtype is not Datatype.uint64:
        raise TypeError(f"mask_low must be uint64type, current type: {mask_low.dtype}")

    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction("set_mask", low=mask_low, high=mask_high)
        )


def reset_mask() -> None:
    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction("reset_mask")
        )
