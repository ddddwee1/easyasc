from typing import Union

from ...utils.Tensor import Tensor
from ...utils.var import Var
from ...utils.datatype import Datatype
from ...utils.positions import Position
from ...utils.instruction import Instruction
from ... import globvars
from .vecutils import (
    infer_repeat,
    infer_repeat_brcb,
    resolve_strides,
    validate_scalar,
    validate_var_or_int,
)


def dup(
    dst: Tensor,
    value: Union[int, float, Var],
    repeat: Union[int, Var, None] = None,
    dst_blk_stride: Union[int, Var, None] = None,
    dst_rep_stride: Union[int, Var, None] = None,
) -> None:
    if not isinstance(dst, Tensor):
        raise TypeError(f"dst必须是Tensor类型，当前类型: {type(dst)}")
    if dst.position is not Position.UB:
        raise ValueError(f"dst必须在UB位置，当前位置: {dst.position}")
    validate_scalar(value, "value")
    if dst.dtype not in (Datatype.float, Datatype.half, Datatype.int):
        raise ValueError(f"不支持的数据类型: {dst.dtype}")

    if repeat is None:
        repeat = infer_repeat(dst)

    dst_blk_stride, dst_rep_stride = resolve_strides(dst, dst_blk_stride, dst_rep_stride)

    validate_var_or_int(repeat, "repeat")
    validate_var_or_int(dst_blk_stride, "dst_blk_stride")
    validate_var_or_int(dst_rep_stride, "dst_rep_stride")

    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction(
                "dup",
                dst=dst,
                src=value,
                repeat=repeat,
                dst_blk_stride=dst_blk_stride,
                dst_rep_stride=dst_rep_stride,
            )
        )


def brcb(
    dst: Tensor,
    src: Tensor,
    dst_blk_stride: Union[int, Var, None] = None,
    dst_rep_stride: Union[int, Var, None] = None,
    repeat: Union[int, Var, None] = None,
) -> None:
    if not isinstance(dst, Tensor):
        raise TypeError(f"dst必须是Tensor类型，当前类型: {type(dst)}")
    if not isinstance(src, Tensor):
        raise TypeError(f"src必须是Tensor类型，当前类型: {type(src)}")
    if dst.position is not Position.UB:
        raise ValueError(f"dst必须在UB位置，当前位置: {dst.position}")
    if src.position is not Position.UB:
        raise ValueError(f"src必须在UB位置，当前位置: {src.position}")
    if dst.dtype != src.dtype:
        raise ValueError("dst/src的数据类型必须一致")

    if repeat is None:
        repeat = infer_repeat_brcb(src)

    dst_blk_stride, dst_rep_stride = resolve_strides(dst, dst_blk_stride, dst_rep_stride)

    validate_var_or_int(repeat, "repeat")
    validate_var_or_int(dst_blk_stride, "dst_blk_stride")
    validate_var_or_int(dst_rep_stride, "dst_rep_stride")

    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction(
                "brcb",
                dst=dst,
                src=src,
                repeat=repeat,
                dst_blk_stride=dst_blk_stride,
                dst_rep_stride=dst_rep_stride,
            )
        )
