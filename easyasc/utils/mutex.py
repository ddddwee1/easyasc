from typing import Optional

from .. import globvars
from ..stub_functions.crosscore import cube_ready, vec_ready, wait_cube, wait_vec
from .pipe import Pipe, PipeType


class CvMutex:
    def __init__(
        self,
        flag_id: int,
        depth: int = 2,
        src_start_pipe: PipeType = Pipe.S,
        dst_start_pipe: PipeType = Pipe.S,
        src_end_pipe: PipeType = Pipe.FIX,
        dst_end_pipe: Optional[PipeType] = None,
    ):
        self.flag_id = flag_id
        self.depth = depth
        self.src_start_pipe = src_start_pipe
        self.dst_start_pipe = dst_start_pipe
        self.src_end_pipe = src_end_pipe
        if dst_end_pipe is None:
            if globvars.device_type.startswith("b"):
                dst_end_pipe = Pipe.MTE2
            else:
                dst_end_pipe = Pipe.MTE3
        self.dst_end_pipe = dst_end_pipe
        if globvars.active_kernel is not None:
            for mutex in globvars.active_kernel.crosscore_mutex:
                if getattr(mutex, "flag_id", None) == flag_id:
                    raise ValueError(f"crosscore_mutex already contains the same flag_id: {flag_id}")
            globvars.active_kernel.crosscore_mutex.append(self)

    def lock(self) -> None:
        wait_vec(self.flag_id, self.src_start_pipe)

    def ready(self) -> None:
        cube_ready(self.flag_id, self.src_end_pipe)

    def wait(self) -> None:
        wait_cube(self.flag_id, self.dst_start_pipe)

    def free(self) -> None:
        vec_ready(self.flag_id, self.dst_end_pipe)


class VcMutex:
    def __init__(
        self,
        flag_id: int,
        depth: int = 2,
        src_start_pipe: PipeType = Pipe.S,
        dst_start_pipe: PipeType = Pipe.S,
        src_end_pipe: PipeType = Pipe.MTE3,
        dst_end_pipe: PipeType = Pipe.FIX,
    ):
        self.flag_id = flag_id
        self.depth = depth
        self.src_start_pipe = src_start_pipe
        self.dst_start_pipe = dst_start_pipe
        self.src_end_pipe = src_end_pipe
        self.dst_end_pipe = dst_end_pipe
        if globvars.active_kernel is not None:
            for mutex in globvars.active_kernel.crosscore_mutex:
                if getattr(mutex, "flag_id", None) == flag_id:
                    raise ValueError(f"crosscore_mutex already contains the same flag_id: {flag_id}")
            globvars.active_kernel.crosscore_mutex.append(self)

    def lock(self) -> None:
        wait_cube(self.flag_id, self.src_start_pipe)

    def ready(self) -> None:
        vec_ready(self.flag_id, self.src_end_pipe)

    def wait(self) -> None:
        wait_vec(self.flag_id, self.dst_start_pipe)

    def free(self) -> None:
        cube_ready(self.flag_id, self.dst_end_pipe)
