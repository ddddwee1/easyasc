from typing import Union, Tuple

from ...utils.Tensor import Tensor
from ...utils.var import Var
from ...utils.datatype import Datatype
from ...utils.positions import Position
from ...utils.instruction import Instruction
from ... import globvars
from .vecutils import infer_repeat, resolve_strides, validate_var_or_int


def _validate_unary_tensors(
    dst: Tensor,
    src: Tensor,
    allowed_dtypes: Tuple,
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
    if dst.dtype not in allowed_dtypes:
        raise ValueError(f"不支持的数据类型: {dst.dtype}")


def _emit_unary_inst(
    opname: str,
    dst: Tensor,
    src: Tensor,
    repeat: Union[int, Var],
    dst_blk_stride: Union[int, Var],
    src_blk_stride: Union[int, Var],
    dst_rep_stride: Union[int, Var],
    src_rep_stride: Union[int, Var],
) -> None:
    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction(
                opname,
                dst=dst,
                src=src,
                repeat=repeat,
                dst_blk_stride=dst_blk_stride,
                src_blk_stride=src_blk_stride,
                dst_rep_stride=dst_rep_stride,
                src_rep_stride=src_rep_stride,
            )
        )


def exp(
    dst: Tensor,
    src: Tensor,
    repeat: Union[int, Var, None] = None,
    dst_blk_stride: Union[int, Var, None] = None,
    src_blk_stride: Union[int, Var, None] = None,
    dst_rep_stride: Union[int, Var, None] = None,
    src_rep_stride: Union[int, Var, None] = None,
) -> None:
    _validate_unary_tensors(dst, src, (Datatype.float, Datatype.half))

    if repeat is None:
        repeat = infer_repeat(src)

    dst_blk_stride, dst_rep_stride = resolve_strides(dst, dst_blk_stride, dst_rep_stride)
    src_blk_stride, src_rep_stride = resolve_strides(src, src_blk_stride, src_rep_stride)

    validate_var_or_int(repeat, "repeat")
    validate_var_or_int(dst_blk_stride, "dst_blk_stride")
    validate_var_or_int(src_blk_stride, "src_blk_stride")
    validate_var_or_int(dst_rep_stride, "dst_rep_stride")
    validate_var_or_int(src_rep_stride, "src_rep_stride")

    _emit_unary_inst(
        "exp",
        dst,
        src,
        repeat,
        dst_blk_stride,
        src_blk_stride,
        dst_rep_stride,
        src_rep_stride,
    )


def ln(
    dst: Tensor,
    src: Tensor,
    repeat: Union[int, Var, None] = None,
    dst_blk_stride: Union[int, Var, None] = None,
    src_blk_stride: Union[int, Var, None] = None,
    dst_rep_stride: Union[int, Var, None] = None,
    src_rep_stride: Union[int, Var, None] = None,
) -> None:
    _validate_unary_tensors(dst, src, (Datatype.float, Datatype.half))

    if repeat is None:
        repeat = infer_repeat(src)

    dst_blk_stride, dst_rep_stride = resolve_strides(dst, dst_blk_stride, dst_rep_stride)
    src_blk_stride, src_rep_stride = resolve_strides(src, src_blk_stride, src_rep_stride)

    validate_var_or_int(repeat, "repeat")
    validate_var_or_int(dst_blk_stride, "dst_blk_stride")
    validate_var_or_int(src_blk_stride, "src_blk_stride")
    validate_var_or_int(dst_rep_stride, "dst_rep_stride")
    validate_var_or_int(src_rep_stride, "src_rep_stride")

    _emit_unary_inst(
        "ln",
        dst,
        src,
        repeat,
        dst_blk_stride,
        src_blk_stride,
        dst_rep_stride,
        src_rep_stride,
    )


def abs(
    dst: Tensor,
    src: Tensor,
    repeat: Union[int, Var, None] = None,
    dst_blk_stride: Union[int, Var, None] = None,
    src_blk_stride: Union[int, Var, None] = None,
    dst_rep_stride: Union[int, Var, None] = None,
    src_rep_stride: Union[int, Var, None] = None,
) -> None:
    _validate_unary_tensors(dst, src, (Datatype.float, Datatype.half))

    if repeat is None:
        repeat = infer_repeat(src)

    dst_blk_stride, dst_rep_stride = resolve_strides(dst, dst_blk_stride, dst_rep_stride)
    src_blk_stride, src_rep_stride = resolve_strides(src, src_blk_stride, src_rep_stride)

    validate_var_or_int(repeat, "repeat")
    validate_var_or_int(dst_blk_stride, "dst_blk_stride")
    validate_var_or_int(src_blk_stride, "src_blk_stride")
    validate_var_or_int(dst_rep_stride, "dst_rep_stride")
    validate_var_or_int(src_rep_stride, "src_rep_stride")

    _emit_unary_inst(
        "abs",
        dst,
        src,
        repeat,
        dst_blk_stride,
        src_blk_stride,
        dst_rep_stride,
        src_rep_stride,
    )


def rec(
    dst: Tensor,
    src: Tensor,
    repeat: Union[int, Var, None] = None,
    dst_blk_stride: Union[int, Var, None] = None,
    src_blk_stride: Union[int, Var, None] = None,
    dst_rep_stride: Union[int, Var, None] = None,
    src_rep_stride: Union[int, Var, None] = None,
) -> None:
    _validate_unary_tensors(dst, src, (Datatype.float, Datatype.half))

    if repeat is None:
        repeat = infer_repeat(src)

    dst_blk_stride, dst_rep_stride = resolve_strides(dst, dst_blk_stride, dst_rep_stride)
    src_blk_stride, src_rep_stride = resolve_strides(src, src_blk_stride, src_rep_stride)

    validate_var_or_int(repeat, "repeat")
    validate_var_or_int(dst_blk_stride, "dst_blk_stride")
    validate_var_or_int(src_blk_stride, "src_blk_stride")
    validate_var_or_int(dst_rep_stride, "dst_rep_stride")
    validate_var_or_int(src_rep_stride, "src_rep_stride")

    _emit_unary_inst(
        "rec",
        dst,
        src,
        repeat,
        dst_blk_stride,
        src_blk_stride,
        dst_rep_stride,
        src_rep_stride,
    )


def sqrt(
    dst: Tensor,
    src: Tensor,
    repeat: Union[int, Var, None] = None,
    dst_blk_stride: Union[int, Var, None] = None,
    src_blk_stride: Union[int, Var, None] = None,
    dst_rep_stride: Union[int, Var, None] = None,
    src_rep_stride: Union[int, Var, None] = None,
) -> None:
    _validate_unary_tensors(dst, src, (Datatype.float, Datatype.half))

    if repeat is None:
        repeat = infer_repeat(src)

    dst_blk_stride, dst_rep_stride = resolve_strides(dst, dst_blk_stride, dst_rep_stride)
    src_blk_stride, src_rep_stride = resolve_strides(src, src_blk_stride, src_rep_stride)

    validate_var_or_int(repeat, "repeat")
    validate_var_or_int(dst_blk_stride, "dst_blk_stride")
    validate_var_or_int(src_blk_stride, "src_blk_stride")
    validate_var_or_int(dst_rep_stride, "dst_rep_stride")
    validate_var_or_int(src_rep_stride, "src_rep_stride")

    _emit_unary_inst(
        "sqrt",
        dst,
        src,
        repeat,
        dst_blk_stride,
        src_blk_stride,
        dst_rep_stride,
        src_rep_stride,
    )


def rsqrt(
    dst: Tensor,
    src: Tensor,
    repeat: Union[int, Var, None] = None,
    dst_blk_stride: Union[int, Var, None] = None,
    src_blk_stride: Union[int, Var, None] = None,
    dst_rep_stride: Union[int, Var, None] = None,
    src_rep_stride: Union[int, Var, None] = None,
) -> None:
    _validate_unary_tensors(dst, src, (Datatype.float, Datatype.half))

    if repeat is None:
        repeat = infer_repeat(src)

    dst_blk_stride, dst_rep_stride = resolve_strides(dst, dst_blk_stride, dst_rep_stride)
    src_blk_stride, src_rep_stride = resolve_strides(src, src_blk_stride, src_rep_stride)

    validate_var_or_int(repeat, "repeat")
    validate_var_or_int(dst_blk_stride, "dst_blk_stride")
    validate_var_or_int(src_blk_stride, "src_blk_stride")
    validate_var_or_int(dst_rep_stride, "dst_rep_stride")
    validate_var_or_int(src_rep_stride, "src_rep_stride")

    _emit_unary_inst(
        "rsqrt",
        dst,
        src,
        repeat,
        dst_blk_stride,
        src_blk_stride,
        dst_rep_stride,
        src_rep_stride,
    )


def vnot(
    dst: Tensor,
    src: Tensor,
    repeat: Union[int, Var, None] = None,
    dst_blk_stride: Union[int, Var, None] = None,
    src_blk_stride: Union[int, Var, None] = None,
    dst_rep_stride: Union[int, Var, None] = None,
    src_rep_stride: Union[int, Var, None] = None,
) -> None:
    _validate_unary_tensors(dst, src, (Datatype.int,))

    if repeat is None:
        repeat = infer_repeat(src)

    dst_blk_stride, dst_rep_stride = resolve_strides(dst, dst_blk_stride, dst_rep_stride)
    src_blk_stride, src_rep_stride = resolve_strides(src, src_blk_stride, src_rep_stride)

    validate_var_or_int(repeat, "repeat")
    validate_var_or_int(dst_blk_stride, "dst_blk_stride")
    validate_var_or_int(src_blk_stride, "src_blk_stride")
    validate_var_or_int(dst_rep_stride, "dst_rep_stride")
    validate_var_or_int(src_rep_stride, "src_rep_stride")

    _emit_unary_inst(
        "vnot",
        dst,
        src,
        repeat,
        dst_blk_stride,
        src_blk_stride,
        dst_rep_stride,
        src_rep_stride,
    )


def relu(
    dst: Tensor,
    src: Tensor,
    repeat: Union[int, Var, None] = None,
    dst_blk_stride: Union[int, Var, None] = None,
    src_blk_stride: Union[int, Var, None] = None,
    dst_rep_stride: Union[int, Var, None] = None,
    src_rep_stride: Union[int, Var, None] = None,
) -> None:
    _validate_unary_tensors(dst, src, (Datatype.float, Datatype.half))

    if repeat is None:
        repeat = infer_repeat(src)

    dst_blk_stride, dst_rep_stride = resolve_strides(dst, dst_blk_stride, dst_rep_stride)
    src_blk_stride, src_rep_stride = resolve_strides(src, src_blk_stride, src_rep_stride)

    validate_var_or_int(repeat, "repeat")
    validate_var_or_int(dst_blk_stride, "dst_blk_stride")
    validate_var_or_int(src_blk_stride, "src_blk_stride")
    validate_var_or_int(dst_rep_stride, "dst_rep_stride")
    validate_var_or_int(src_rep_stride, "src_rep_stride")

    _emit_unary_inst(
        "relu",
        dst,
        src,
        repeat,
        dst_blk_stride,
        src_blk_stride,
        dst_rep_stride,
        src_rep_stride,
    )
