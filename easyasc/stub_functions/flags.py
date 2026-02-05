from typing import Union

from ..utils.pipe import PipeType
from ..utils.var import Var
from ..utils.instruction import Instruction
from .. import globvars


def setflag(src: PipeType, dst: PipeType, event_id: Union[int, Var]) -> None:
    if not isinstance(src, PipeType):
        raise TypeError(f"src必须是PipeType类型，当前类型: {type(src)}")
    if not isinstance(dst, PipeType):
        raise TypeError(f"dst必须是PipeType类型，当前类型: {type(dst)}")
    if not isinstance(event_id, (int, Var)):
        raise TypeError(f"event_id必须是Var或int类型，当前类型: {type(event_id)}")
    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction("setflag", src=src, dst=dst, event_id=event_id)
        )


def waitflag(src: PipeType, dst: PipeType, event_id: Union[int, Var]) -> None:
    if not isinstance(src, PipeType):
        raise TypeError(f"src必须是PipeType类型，当前类型: {type(src)}")
    if not isinstance(dst, PipeType):
        raise TypeError(f"dst必须是PipeType类型，当前类型: {type(dst)}")
    if not isinstance(event_id, (int, Var)):
        raise TypeError(f"event_id必须是Var或int类型，当前类型: {type(event_id)}")
    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction("waitflag", src=src, dst=dst, event_id=event_id)
        )
