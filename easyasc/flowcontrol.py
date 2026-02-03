from typing import Iterator, Union, Optional

from .utils.var import Var
from .utils.instruction import Instruction
from .kernelbase.kernelbase import KernelBase
from . import globvars


def unroll(*args):
    return range(*args)


def _validate_range_arg(value: Union[Var, int], label: str) -> None:
    if not isinstance(value, (int, Var)):
        raise TypeError(f"{label} must be int or Var, got: {type(value)}")


def range(*args: Union[Var, int], name: str = "") -> Iterator[Var]:
    if not isinstance(name, str):
        raise TypeError(f"name must be str, got: {type(name)}")

    if len(args) == 1:
        start: Union[Var, int] = 0
        stop: Union[Var, int] = args[0]
        step: Union[Var, int] = 1
    elif len(args) == 2:
        start = args[0]
        stop = args[1]
        step = 1
    elif len(args) == 3:
        start, stop, step = args
    else:
        raise TypeError("range expected 1-3 positional arguments")

    _validate_range_arg(start, "start")
    _validate_range_arg(stop, "stop")
    _validate_range_arg(step, "step")
    if isinstance(step, int) and step == 0:
        raise ValueError("range() arg 3 must not be zero")
    if isinstance(step, Var) and isinstance(step.value, int) and step.value == 0:
        raise ValueError("range() arg 3 must not be zero")

    if isinstance(start, Var):
        start_value = start.value if not isinstance(start.value, Var) else None
        loop_var = Var(value=start_value, dtype=start.dtype, name=name)
    else:
        loop_var = Var(value=start, name=name)

    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction("start_loop", var=loop_var, start=start, stop=stop, step=step)
        )

    try:
        yield loop_var
    finally:
        if globvars.active_kernel is not None:
            globvars.active_kernel.instructions.append(
                Instruction("end_loop", var=loop_var)
            )


def _pop_last_end_if() -> None:
    if globvars.active_kernel is None:
        raise RuntimeError("If/Elif/Else只能在kernel内使用")
    if not globvars.active_kernel.instructions:
        raise RuntimeError("Elif/Else必须跟在If/Elif之后")
    last = globvars.active_kernel.instructions[-1]
    if last.opname != "end_if":
        raise RuntimeError("Elif/Else必须跟在If/Elif之后")
    globvars.active_kernel.instructions.pop()


class If:
    def __init__(self, cond):
        self.cond = cond

    def __enter__(self):
        if globvars.active_kernel is None:
            raise RuntimeError("If只能在kernel内使用")
        globvars.active_kernel.instructions.append(
            Instruction("start_if", cond=self.cond)
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            return False
        if globvars.active_kernel is None:
            raise RuntimeError("If只能在kernel内使用")
        globvars.active_kernel.instructions.append(
            Instruction("end_if")
        )
        return True


class Elif:
    def __init__(self, cond):
        self.cond = cond

    def __enter__(self):
        if not isinstance(globvars.active_kernel, KernelBase):
            raise RuntimeError("Elif can only be used inside kernel")
        _pop_last_end_if()
        globvars.active_kernel.instructions.append(
            Instruction("start_elif", cond=self.cond)
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            return False
        if globvars.active_kernel is None:
            raise RuntimeError("Elif只能在kernel内使用")
        globvars.active_kernel.instructions.append(
            Instruction("end_if")
        )
        return True


class Else:
    def __enter__(self):
        if not isinstance(globvars.active_kernel, KernelBase):
            raise RuntimeError("Else can only be used inside kernel")
        _pop_last_end_if()
        globvars.active_kernel.instructions.append(
            Instruction("start_else")
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            return False
        if globvars.active_kernel is None:
            raise RuntimeError("Else只能在kernel内使用")
        globvars.active_kernel.instructions.append(
            Instruction("end_if")
        )
        return True
