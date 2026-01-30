from ..utils.instruction import Instruction
from ..utils.pipe import PipeType, Pipe
from .. import globvars


def cube_ready(flag_id: int = 0, pipe: PipeType = Pipe.FIX) -> None:
    if not isinstance(flag_id, int):
        raise TypeError(f"flag_id必须是int类型，当前类型: {type(flag_id)}")
    if not isinstance(pipe, PipeType):
        raise TypeError(f"pipe必须是PipeType类型，当前类型: {type(pipe)}")
    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction("cube_ready", flag_id=flag_id, pipe=pipe)
        )


def vec_ready(flag_id: int = 0, pipe: PipeType = Pipe.MTE3) -> None:
    if not isinstance(flag_id, int):
        raise TypeError(f"flag_id必须是int类型，当前类型: {type(flag_id)}")
    if not isinstance(pipe, PipeType):
        raise TypeError(f"pipe必须是PipeType类型，当前类型: {type(pipe)}")
    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction("vec_ready", flag_id=flag_id, pipe=pipe)
        )


def wait_cube(flag_id: int = 0, pipe: PipeType = Pipe.S) -> None:
    if not isinstance(flag_id, int):
        raise TypeError(f"flag_id必须是int类型，当前类型: {type(flag_id)}")
    if not isinstance(pipe, PipeType):
        raise TypeError(f"pipe必须是PipeType类型，当前类型: {type(pipe)}")
    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction("wait_cube", flag_id=flag_id, pipe=pipe)
        )


def wait_vec(flag_id: int = 0, pipe: PipeType = Pipe.S) -> None:
    if not isinstance(flag_id, int):
        raise TypeError(f"flag_id必须是int类型，当前类型: {type(flag_id)}")
    if not isinstance(pipe, PipeType):
        raise TypeError(f"pipe必须是PipeType类型，当前类型: {type(pipe)}")
    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction("wait_vec", flag_id=flag_id, pipe=pipe)
        )


def allcube_ready(flag_id: int = 0, pipe: PipeType = Pipe.FIX) -> None:
    if not isinstance(flag_id, int):
        raise TypeError(f"flag_id必须是int类型，当前类型: {type(flag_id)}")
    if not isinstance(pipe, PipeType):
        raise TypeError(f"pipe必须是PipeType类型，当前类型: {type(pipe)}")
    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction("allcube_ready", flag_id=flag_id, pipe=pipe)
        )


def allvec_ready(flag_id: int = 0, pipe: PipeType = Pipe.MTE3) -> None:
    if not isinstance(flag_id, int):
        raise TypeError(f"flag_id必须是int类型，当前类型: {type(flag_id)}")
    if not isinstance(pipe, PipeType):
        raise TypeError(f"pipe必须是PipeType类型，当前类型: {type(pipe)}")
    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction("allvec_ready", flag_id=flag_id, pipe=pipe)
        )


def allcube_wait(flag_id: int = 0, pipe: PipeType = Pipe.S) -> None:
    if not isinstance(flag_id, int):
        raise TypeError(f"flag_id必须是int类型，当前类型: {type(flag_id)}")
    if not isinstance(pipe, PipeType):
        raise TypeError(f"pipe必须是PipeType类型，当前类型: {type(pipe)}")
    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction("allcube_wait", flag_id=flag_id, pipe=pipe)
        )


def allvec_wait(flag_id: int = 0, pipe: PipeType = Pipe.S) -> None:
    if not isinstance(flag_id, int):
        raise TypeError(f"flag_id必须是int类型，当前类型: {type(flag_id)}")
    if not isinstance(pipe, PipeType):
        raise TypeError(f"pipe必须是PipeType类型，当前类型: {type(pipe)}")
    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction("allvec_wait", flag_id=flag_id, pipe=pipe)
        )
