from .instruction import Instruction
from .pipe import PipeType
from .. import globvars


class _BaseEvent:
    def __init__(self, src_pipe: PipeType, dst_pipe: PipeType, preset: bool, name: str, prefix: str, create_op: str):
        if not isinstance(src_pipe, PipeType):
            raise TypeError(f"src_pipe must be PipeType, got: {type(src_pipe)}")
        if not isinstance(dst_pipe, PipeType):
            raise TypeError(f"dst_pipe must be PipeType, got: {type(dst_pipe)}")
        if not isinstance(preset, bool):
            raise TypeError(f"preset must be bool, got: {type(preset)}")
        if not isinstance(name, str):
            raise TypeError(f"name must be str, got: {type(name)}")

        idx = globvars.tmp_idx
        globvars.tmp_idx += 1
        if name == "":
            name = f"{prefix}{idx}"

        self.src_pipe = src_pipe
        self.dst_pipe = dst_pipe
        self.preset = preset
        self.name = name
        self.idx = idx

        if globvars.active_kernel is not None:
            globvars.active_kernel.instructions.append(
                Instruction(create_op, val=self)
            )

    def __repr__(self):
        return (
            f"{type(self).__name__}(name={self.name!r}, src_pipe={self.src_pipe!r}, "
            f"dst_pipe={self.dst_pipe!r}, preset={self.preset!r}, idx={self.idx!r})"
        )

    def set(self) -> None:
        if globvars.active_kernel is not None:
            globvars.active_kernel.instructions.append(
                Instruction("event_set", event=self)
            )

    def wait(self) -> None:
        if globvars.active_kernel is not None:
            globvars.active_kernel.instructions.append(
                Instruction("event_wait", event=self)
            )

    def setall(self) -> None:
        if globvars.active_kernel is not None:
            globvars.active_kernel.instructions.append(
                Instruction("event_setall", event=self)
            )

    def release(self) -> None:
        if globvars.active_kernel is not None:
            globvars.active_kernel.instructions.append(
                Instruction("event_release", event=self)
            )


class SEvent(_BaseEvent):
    """Single event class."""
    def __init__(self, src_pipe: PipeType, dst_pipe: PipeType, preset: bool = False, name: str = ""):
        super().__init__(src_pipe, dst_pipe, preset, name, "_tmp_sevent_", "create_sevent")


class DEvent(_BaseEvent):
    """Double event class."""
    def __init__(self, src_pipe: PipeType, dst_pipe: PipeType, preset: bool = False, name: str = ""):
        super().__init__(src_pipe, dst_pipe, preset, name, "_tmp_devent_", "create_devent")
