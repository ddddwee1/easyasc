from typing import Iterator, Union, Optional, Any

from .utils.var import Var
from .utils.instruction import Instruction
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

    target = None
    start_opname = "start_loop"
    if globvars.active_micro is not None:
        target = globvars.active_micro
        start_opname = "start_micro_loop"
    elif globvars.active_kernel is not None:
        target = globvars.active_kernel

    if target is not None:
        target.instructions.append(
            Instruction(start_opname, var=loop_var, start=start, stop=stop, step=step)
        )

    try:
        yield loop_var
    finally:
        if target is not None:
            target.instructions.append(
                Instruction("end_loop", var=loop_var)
            )


def _pop_last_end_if() -> None:
    target = _get_flow_target()
    if target is None:
        raise RuntimeError("If/Elif/Else只能在kernel或micro内使用")
    if not target.instructions:
        raise RuntimeError("Elif/Else必须跟在If/Elif之后")
    last = target.instructions[-1]
    if last.opname != "end_if":
        raise RuntimeError("Elif/Else必须跟在If/Elif之后")
    target.instructions.pop()


def _get_flow_target() -> Optional[Any]:
    if globvars.active_micro is not None:
        return globvars.active_micro
    if globvars.active_kernel is not None:
        return globvars.active_kernel
    return None


def _append_flow_instruction(opname: str, **kwargs: Any) -> None:
    target = _get_flow_target()
    if target is None:
        raise RuntimeError("If/Elif/Else只能在kernel或micro内使用")
    target.instructions.append(Instruction(opname, **kwargs))


class If:
    def __init__(self, cond):
        self.cond = cond

    def __enter__(self):
        _append_flow_instruction("start_if", cond=self.cond)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            return False
        _append_flow_instruction("end_if")
        return True


class Elif:
    def __init__(self, cond):
        self.cond = cond

    def __enter__(self):
        if _get_flow_target() is None:
            raise RuntimeError("Elif只能在kernel或micro内使用")
        _pop_last_end_if()
        _append_flow_instruction("start_elif", cond=self.cond)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            return False
        _append_flow_instruction("end_if")
        return True


class Else:
    def __enter__(self):
        if _get_flow_target() is None:
            raise RuntimeError("Else只能在kernel或micro内使用")
        _pop_last_end_if()
        _append_flow_instruction("start_else")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            return False
        _append_flow_instruction("end_if")
        return True
