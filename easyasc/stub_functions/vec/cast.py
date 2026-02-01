from typing import Union

from ...utils.Tensor import Tensor
from ...utils.var import Var
from ...utils.datatype import Datatype
from ...utils.positions import Position
from ...utils.instruction import Instruction
from ...utils.roundmode import RoundMode, RoundModeType
from ... import globvars
from .vecutils import infer_repeat, resolve_strides, validate_var_or_int


def _infer_cast_repeat(dst: Tensor, src: Tensor) -> Union[int, Var]:
    span = dst.span if hasattr(dst, "span") else dst.shape
    dim0 = span[0]
    dim1 = span[1]
    validate_var_or_int(dim0, "dst.span[0]")
    validate_var_or_int(dim1, "dst.span[1]")
    count = dim0 * dim1
    denom = 256 // max(src.dtype.size, dst.dtype.size)
    if isinstance(count, Var):
        return count // denom
    if not isinstance(count, int):
        raise TypeError(f"repeat无法由shape推断，当前count类型: {type(count)}")
    return count // denom


def _resolve_cast_rep_strides(
    dst: Tensor,
    src: Tensor,
    dst_rep_stride: Union[int, Var, None],
    src_rep_stride: Union[int, Var, None],
) -> tuple[Union[int, Var], Union[int, Var]]:
    if (dst_rep_stride is None) != (src_rep_stride is None):
        raise ValueError("src_rep_stride和dst_rep_stride必须同时为None或同时指定")
    if dst_rep_stride is not None and src_rep_stride is not None:
        return dst_rep_stride, src_rep_stride

    c0_src = src.dtype.C0
    c0_dst = dst.dtype.C0
    if c0_src > c0_dst:
        src_rep_stride = 8 * c0_dst // c0_src
        dst_rep_stride = 8 
    elif c0_src < c0_dst:
        dst_rep_stride = 8 * c0_src // c0_dst
        src_rep_stride = 8 
    else:
        src_rep_stride = 8
        dst_rep_stride = 8
    return dst_rep_stride, src_rep_stride


def cast(
    dst: Tensor,
    src: Tensor,
    repeat: Union[int, Var, None] = None,
    dst_blk_stride: Union[int, Var, None] = None,
    src_blk_stride: Union[int, Var, None] = None,
    dst_rep_stride: Union[int, Var, None] = None,
    src_rep_stride: Union[int, Var, None] = None,
    round_mode: RoundModeType = RoundMode.AWAY_FROM_ZERO,
) -> None:
    if not isinstance(dst, Tensor):
        raise TypeError(f"dst必须是Tensor类型，当前类型: {type(dst)}")
    if not isinstance(src, Tensor):
        raise TypeError(f"src必须是Tensor类型，当前类型: {type(src)}")
    if dst.position is not Position.UB:
        raise ValueError(f"dst必须在UB位置，当前位置: {dst.position}")
    if src.position is not Position.UB:
        raise ValueError(f"src必须在UB位置，当前位置: {src.position}")
    if not isinstance(round_mode, RoundModeType):
        raise TypeError(f"round_mode必须是RoundModeType类型，当前类型: {type(round_mode)}")

    if repeat is None:
        repeat = _infer_cast_repeat(dst, src)

    if dst_blk_stride is None or src_blk_stride is None:
        auto_dst_blk, _ = resolve_strides(dst, dst_blk_stride, 8)
        auto_src_blk, _ = resolve_strides(src, src_blk_stride, 8)
        if dst_blk_stride is None:
            dst_blk_stride = auto_dst_blk
        if src_blk_stride is None:
            src_blk_stride = auto_src_blk

    dst_rep_stride, src_rep_stride = _resolve_cast_rep_strides(
        dst, src, dst_rep_stride, src_rep_stride
    )

    if dst.dtype.size > src.dtype.size:
        round_mode = RoundMode.NONE

    validate_var_or_int(repeat, "repeat")
    validate_var_or_int(dst_blk_stride, "dst_blk_stride")
    validate_var_or_int(src_blk_stride, "src_blk_stride")
    validate_var_or_int(dst_rep_stride, "dst_rep_stride")
    validate_var_or_int(src_rep_stride, "src_rep_stride")

    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction(
                "cast",
                dst=dst,
                src=src,
                mode=round_mode,
                repeat=repeat,
                dst_blk_stride=dst_blk_stride,
                src_blk_stride=src_blk_stride,
                dst_rep_stride=dst_rep_stride,
                src_rep_stride=src_rep_stride,
            )
        )
