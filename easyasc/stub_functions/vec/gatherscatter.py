from typing import Union

from ...utils.Tensor import Tensor
from ...utils.var import Var
from ...utils.datatype import Datatype
from ...utils.positions import Position
from ...utils.instruction import Instruction
from ... import globvars
from .vecutils import infer_repeat, resolve_strides, validate_var_or_int


def gather(
    dst: Tensor,
    src: Tensor,
    offset: Tensor,
    repeat: Union[int, Var, None] = None,
    dst_rep_stride: Union[int, Var, None] = None,
) -> None:
    if not isinstance(dst, Tensor):
        raise TypeError(f"dst must be Tensor type, current type: {type(dst)}")
    if not isinstance(src, Tensor):
        raise TypeError(f"src must be Tensor type, current type: {type(src)}")
    if not isinstance(offset, Tensor):
        raise TypeError(f"offset must be Tensor type, current type: {type(offset)}")
    if dst.position is not Position.UB:
        raise ValueError(f"dst must be at UB position, current position: {dst.position}")
    if src.position is not Position.UB:
        raise ValueError(f"src must be at UB position, current position: {src.position}")
    if offset.position is not Position.UB:
        raise ValueError(f"offset must be at UB position, current position: {offset.position}")
    if dst.dtype != src.dtype:
        raise ValueError("dst/src data types must match")
    if offset.dtype is not Datatype.uint32:
        raise ValueError(f"offset must be uint32 type, current type: {offset.dtype}") 
    if repeat is None:
        repeat = infer_repeat(dst)
    if dst_rep_stride is None:
        _, dst_rep_stride = resolve_strides(dst, None, None)

    validate_var_or_int(repeat, "repeat")
    validate_var_or_int(dst_rep_stride, "dst_rep_stride")

    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction(
                "gather",
                dst=dst,
                src=src,
                offset=offset,
                repeat=repeat,
                dst_rep_stride=dst_rep_stride,
            )
        )


def scatter(
    dst: Tensor,
    src: Tensor,
    offset: Tensor,
    repeat: Union[int, Var, None] = None,
    src_rep_stride: Union[int, Var, None] = None,
) -> None:
    if not isinstance(dst, Tensor):
        raise TypeError(f"dst must be Tensor type, current type: {type(dst)}")
    if not isinstance(src, Tensor):
        raise TypeError(f"src must be Tensor type, current type: {type(src)}")
    if not isinstance(offset, Tensor):
        raise TypeError(f"offset must be Tensor type, current type: {type(offset)}")
    if dst.position is not Position.UB:
        raise ValueError(f"dst must be at UB position, current position: {dst.position}")
    if src.position is not Position.UB:
        raise ValueError(f"src must be at UB position, current position: {src.position}")
    if offset.position is not Position.UB:
        raise ValueError(f"offset must be at UB position, current position: {offset.position}")
    if dst.dtype != src.dtype:
        raise ValueError("dst/src data types must match")
    if offset.dtype is not Datatype.uint32:
        raise ValueError(f"offset must be uint32 type, current type: {offset.dtype}") 
    if repeat is None:
        repeat = infer_repeat(src)
    if src_rep_stride is None:
        _, src_rep_stride = resolve_strides(src, None, None)

    validate_var_or_int(repeat, "repeat")
    validate_var_or_int(src_rep_stride, "src_rep_stride")

    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction(
                "scatter",
                dst=dst,
                src=src,
                offset=offset,
                repeat=repeat,
                src_rep_stride=src_rep_stride,
            )
        )
