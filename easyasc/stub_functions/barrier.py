from ..utils.instruction import Instruction
from ..utils.pipe import PipeType, Pipe
from .. import globvars


def barrier(pipe: PipeType) -> None:
    if not isinstance(pipe, PipeType):
        raise TypeError(f"pipe必须是PipeType类型，当前类型: {type(pipe)}")
    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction("barrier", pipe=pipe)
        )


def bar_m() -> None:
    barrier(Pipe.M)


def bar_v() -> None:
    barrier(Pipe.V)


def bar_mte3() -> None:
    barrier(Pipe.MTE3)


def bar_mte2() -> None:
    barrier(Pipe.MTE2)


def bar_mte1() -> None:
    barrier(Pipe.MTE1)


def bar_fix() -> None:
    barrier(Pipe.FIX)


def bar_all() -> None:
    barrier(Pipe.ALL)
