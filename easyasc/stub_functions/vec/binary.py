from typing import Union, Tuple

from ...utils.Tensor import Tensor
from ...utils.var import Var
from ...utils.datatype import Datatype
from ...utils.positions import Position
from ...utils.instruction import Instruction
from ... import globvars
from .vecutils import infer_repeat, resolve_strides, validate_var_or_int


def _validate_binary_tensors(
    dst: Tensor,
    src1: Tensor,
    src2: Tensor,
    allowed_dtypes: Tuple,
) -> None:
    if not isinstance(dst, Tensor):
        raise TypeError(f"dst必须是Tensor类型，当前类型: {type(dst)}")
    if not isinstance(src1, Tensor):
        raise TypeError(f"src1必须是Tensor类型，当前类型: {type(src1)}")
    if not isinstance(src2, Tensor):
        raise TypeError(f"src2必须是Tensor类型，当前类型: {type(src2)}")

    if dst.position is not Position.UB:
        raise ValueError(f"dst必须在UB位置，当前位置: {dst.position}")
    if src1.position is not Position.UB:
        raise ValueError(f"src1必须在UB位置，当前位置: {src1.position}")
    if src2.position is not Position.UB:
        raise ValueError(f"src2必须在UB位置，当前位置: {src2.position}")

    if dst.dtype != src1.dtype or dst.dtype != src2.dtype:
        raise ValueError("dst/src1/src2的数据类型必须一致")
    if dst.dtype not in allowed_dtypes:
        raise ValueError(f"不支持的数据类型: {dst.dtype}")


def _emit_binary_inst(
    opname: str,
    dst: Tensor,
    src1: Tensor,
    src2: Tensor,
    repeat: Union[int, Var],
    dst_blk_stride: Union[int, Var],
    src1_blk_stride: Union[int, Var],
    src2_blk_stride: Union[int, Var],
    dst_rep_stride: Union[int, Var],
    src1_rep_stride: Union[int, Var],
    src2_rep_stride: Union[int, Var],
) -> None:
    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction(
                opname,
                dst=dst,
                src1=src1,
                src2=src2,
                repeat=repeat,
                dst_blk_stride=dst_blk_stride,
                src1_blk_stride=src1_blk_stride,
                src2_blk_stride=src2_blk_stride,
                dst_rep_stride=dst_rep_stride,
                src1_rep_stride=src1_rep_stride,
                src2_rep_stride=src2_rep_stride,
            )
        )


def add(
    dst: Tensor,
    src1: Tensor,
    src2: Tensor,
    repeat: Union[int, Var, None] = None,
    dst_blk_stride: Union[int, Var, None] = None,
    src1_blk_stride: Union[int, Var, None] = None,
    src2_blk_stride: Union[int, Var, None] = None,
    dst_rep_stride: Union[int, Var, None] = None,
    src1_rep_stride: Union[int, Var, None] = None,
    src2_rep_stride: Union[int, Var, None] = None,
) -> None:
    _validate_binary_tensors(dst, src1, src2, (Datatype.float, Datatype.half, Datatype.int))

    if repeat is None:
        repeat = infer_repeat(dst)

    dst_blk_stride, dst_rep_stride = resolve_strides(dst, dst_blk_stride, dst_rep_stride)
    src1_blk_stride, src1_rep_stride = resolve_strides(src1, src1_blk_stride, src1_rep_stride)
    src2_blk_stride, src2_rep_stride = resolve_strides(src2, src2_blk_stride, src2_rep_stride)

    validate_var_or_int(repeat, "repeat")
    validate_var_or_int(dst_blk_stride, "dst_blk_stride")
    validate_var_or_int(src1_blk_stride, "src1_blk_stride")
    validate_var_or_int(src2_blk_stride, "src2_blk_stride")
    validate_var_or_int(dst_rep_stride, "dst_rep_stride")
    validate_var_or_int(src1_rep_stride, "src1_rep_stride")
    validate_var_or_int(src2_rep_stride, "src2_rep_stride")

    _emit_binary_inst(
        "ADD",
        dst,
        src1,
        src2,
        repeat,
        dst_blk_stride,
        src1_blk_stride,
        src2_blk_stride,
        dst_rep_stride,
        src1_rep_stride,
        src2_rep_stride,
    )


def sub(
    dst: Tensor,
    src1: Tensor,
    src2: Tensor,
    repeat: Union[int, Var, None] = None,
    dst_blk_stride: Union[int, Var, None] = None,
    src1_blk_stride: Union[int, Var, None] = None,
    src2_blk_stride: Union[int, Var, None] = None,
    dst_rep_stride: Union[int, Var, None] = None,
    src1_rep_stride: Union[int, Var, None] = None,
    src2_rep_stride: Union[int, Var, None] = None,
) -> None:
    _validate_binary_tensors(dst, src1, src2, (Datatype.float, Datatype.half, Datatype.int))

    if repeat is None:
        repeat = infer_repeat(dst)

    dst_blk_stride, dst_rep_stride = resolve_strides(dst, dst_blk_stride, dst_rep_stride)
    src1_blk_stride, src1_rep_stride = resolve_strides(src1, src1_blk_stride, src1_rep_stride)
    src2_blk_stride, src2_rep_stride = resolve_strides(src2, src2_blk_stride, src2_rep_stride)

    validate_var_or_int(repeat, "repeat")
    validate_var_or_int(dst_blk_stride, "dst_blk_stride")
    validate_var_or_int(src1_blk_stride, "src1_blk_stride")
    validate_var_or_int(src2_blk_stride, "src2_blk_stride")
    validate_var_or_int(dst_rep_stride, "dst_rep_stride")
    validate_var_or_int(src1_rep_stride, "src1_rep_stride")
    validate_var_or_int(src2_rep_stride, "src2_rep_stride")

    _emit_binary_inst(
        "SUB",
        dst,
        src1,
        src2,
        repeat,
        dst_blk_stride,
        src1_blk_stride,
        src2_blk_stride,
        dst_rep_stride,
        src1_rep_stride,
        src2_rep_stride,
    )


def mul(
    dst: Tensor,
    src1: Tensor,
    src2: Tensor,
    repeat: Union[int, Var, None] = None,
    dst_blk_stride: Union[int, Var, None] = None,
    src1_blk_stride: Union[int, Var, None] = None,
    src2_blk_stride: Union[int, Var, None] = None,
    dst_rep_stride: Union[int, Var, None] = None,
    src1_rep_stride: Union[int, Var, None] = None,
    src2_rep_stride: Union[int, Var, None] = None,
) -> None:
    _validate_binary_tensors(dst, src1, src2, (Datatype.float, Datatype.half, Datatype.int))

    if repeat is None:
        repeat = infer_repeat(dst)

    dst_blk_stride, dst_rep_stride = resolve_strides(dst, dst_blk_stride, dst_rep_stride)
    src1_blk_stride, src1_rep_stride = resolve_strides(src1, src1_blk_stride, src1_rep_stride)
    src2_blk_stride, src2_rep_stride = resolve_strides(src2, src2_blk_stride, src2_rep_stride)

    validate_var_or_int(repeat, "repeat")
    validate_var_or_int(dst_blk_stride, "dst_blk_stride")
    validate_var_or_int(src1_blk_stride, "src1_blk_stride")
    validate_var_or_int(src2_blk_stride, "src2_blk_stride")
    validate_var_or_int(dst_rep_stride, "dst_rep_stride")
    validate_var_or_int(src1_rep_stride, "src1_rep_stride")
    validate_var_or_int(src2_rep_stride, "src2_rep_stride")

    _emit_binary_inst(
        "MUL",
        dst,
        src1,
        src2,
        repeat,
        dst_blk_stride,
        src1_blk_stride,
        src2_blk_stride,
        dst_rep_stride,
        src1_rep_stride,
        src2_rep_stride,
    )


def div(
    dst: Tensor,
    src1: Tensor,
    src2: Tensor,
    repeat: Union[int, Var, None] = None,
    dst_blk_stride: Union[int, Var, None] = None,
    src1_blk_stride: Union[int, Var, None] = None,
    src2_blk_stride: Union[int, Var, None] = None,
    dst_rep_stride: Union[int, Var, None] = None,
    src1_rep_stride: Union[int, Var, None] = None,
    src2_rep_stride: Union[int, Var, None] = None,
) -> None:
    _validate_binary_tensors(dst, src1, src2, (Datatype.float, Datatype.half))

    if repeat is None:
        repeat = infer_repeat(dst)

    dst_blk_stride, dst_rep_stride = resolve_strides(dst, dst_blk_stride, dst_rep_stride)
    src1_blk_stride, src1_rep_stride = resolve_strides(src1, src1_blk_stride, src1_rep_stride)
    src2_blk_stride, src2_rep_stride = resolve_strides(src2, src2_blk_stride, src2_rep_stride)

    validate_var_or_int(repeat, "repeat")
    validate_var_or_int(dst_blk_stride, "dst_blk_stride")
    validate_var_or_int(src1_blk_stride, "src1_blk_stride")
    validate_var_or_int(src2_blk_stride, "src2_blk_stride")
    validate_var_or_int(dst_rep_stride, "dst_rep_stride")
    validate_var_or_int(src1_rep_stride, "src1_rep_stride")
    validate_var_or_int(src2_rep_stride, "src2_rep_stride")

    _emit_binary_inst(
        "DIV",
        dst,
        src1,
        src2,
        repeat,
        dst_blk_stride,
        src1_blk_stride,
        src2_blk_stride,
        dst_rep_stride,
        src1_rep_stride,
        src2_rep_stride,
    )


def vmax(
    dst: Tensor,
    src1: Tensor,
    src2: Tensor,
    repeat: Union[int, Var, None] = None,
    dst_blk_stride: Union[int, Var, None] = None,
    src1_blk_stride: Union[int, Var, None] = None,
    src2_blk_stride: Union[int, Var, None] = None,
    dst_rep_stride: Union[int, Var, None] = None,
    src1_rep_stride: Union[int, Var, None] = None,
    src2_rep_stride: Union[int, Var, None] = None,
) -> None:
    _validate_binary_tensors(dst, src1, src2, (Datatype.float, Datatype.half, Datatype.int))

    if repeat is None:
        repeat = infer_repeat(dst)

    dst_blk_stride, dst_rep_stride = resolve_strides(dst, dst_blk_stride, dst_rep_stride)
    src1_blk_stride, src1_rep_stride = resolve_strides(src1, src1_blk_stride, src1_rep_stride)
    src2_blk_stride, src2_rep_stride = resolve_strides(src2, src2_blk_stride, src2_rep_stride)

    validate_var_or_int(repeat, "repeat")
    validate_var_or_int(dst_blk_stride, "dst_blk_stride")
    validate_var_or_int(src1_blk_stride, "src1_blk_stride")
    validate_var_or_int(src2_blk_stride, "src2_blk_stride")
    validate_var_or_int(dst_rep_stride, "dst_rep_stride")
    validate_var_or_int(src1_rep_stride, "src1_rep_stride")
    validate_var_or_int(src2_rep_stride, "src2_rep_stride")

    _emit_binary_inst(
        "MAX",
        dst,
        src1,
        src2,
        repeat,
        dst_blk_stride,
        src1_blk_stride,
        src2_blk_stride,
        dst_rep_stride,
        src1_rep_stride,
        src2_rep_stride,
    )


def vmin(
    dst: Tensor,
    src1: Tensor,
    src2: Tensor,
    repeat: Union[int, Var, None] = None,
    dst_blk_stride: Union[int, Var, None] = None,
    src1_blk_stride: Union[int, Var, None] = None,
    src2_blk_stride: Union[int, Var, None] = None,
    dst_rep_stride: Union[int, Var, None] = None,
    src1_rep_stride: Union[int, Var, None] = None,
    src2_rep_stride: Union[int, Var, None] = None,
) -> None:
    _validate_binary_tensors(dst, src1, src2, (Datatype.float, Datatype.half, Datatype.int))

    if repeat is None:
        repeat = infer_repeat(dst)

    dst_blk_stride, dst_rep_stride = resolve_strides(dst, dst_blk_stride, dst_rep_stride)
    src1_blk_stride, src1_rep_stride = resolve_strides(src1, src1_blk_stride, src1_rep_stride)
    src2_blk_stride, src2_rep_stride = resolve_strides(src2, src2_blk_stride, src2_rep_stride)

    validate_var_or_int(repeat, "repeat")
    validate_var_or_int(dst_blk_stride, "dst_blk_stride")
    validate_var_or_int(src1_blk_stride, "src1_blk_stride")
    validate_var_or_int(src2_blk_stride, "src2_blk_stride")
    validate_var_or_int(dst_rep_stride, "dst_rep_stride")
    validate_var_or_int(src1_rep_stride, "src1_rep_stride")
    validate_var_or_int(src2_rep_stride, "src2_rep_stride")

    _emit_binary_inst(
        "MIN",
        dst,
        src1,
        src2,
        repeat,
        dst_blk_stride,
        src1_blk_stride,
        src2_blk_stride,
        dst_rep_stride,
        src1_rep_stride,
        src2_rep_stride,
    )


def vand(
    dst: Tensor,
    src1: Tensor,
    src2: Tensor,
    repeat: Union[int, Var, None] = None,
    dst_blk_stride: Union[int, Var, None] = None,
    src1_blk_stride: Union[int, Var, None] = None,
    src2_blk_stride: Union[int, Var, None] = None,
    dst_rep_stride: Union[int, Var, None] = None,
    src1_rep_stride: Union[int, Var, None] = None,
    src2_rep_stride: Union[int, Var, None] = None,
) -> None:
    _validate_binary_tensors(dst, src1, src2, (Datatype.int,))

    if repeat is None:
        repeat = infer_repeat(dst)

    dst_blk_stride, dst_rep_stride = resolve_strides(dst, dst_blk_stride, dst_rep_stride)
    src1_blk_stride, src1_rep_stride = resolve_strides(src1, src1_blk_stride, src1_rep_stride)
    src2_blk_stride, src2_rep_stride = resolve_strides(src2, src2_blk_stride, src2_rep_stride)

    validate_var_or_int(repeat, "repeat")
    validate_var_or_int(dst_blk_stride, "dst_blk_stride")
    validate_var_or_int(src1_blk_stride, "src1_blk_stride")
    validate_var_or_int(src2_blk_stride, "src2_blk_stride")
    validate_var_or_int(dst_rep_stride, "dst_rep_stride")
    validate_var_or_int(src1_rep_stride, "src1_rep_stride")
    validate_var_or_int(src2_rep_stride, "src2_rep_stride")

    _emit_binary_inst(
        "AND",
        dst,
        src1,
        src2,
        repeat,
        dst_blk_stride,
        src1_blk_stride,
        src2_blk_stride,
        dst_rep_stride,
        src1_rep_stride,
        src2_rep_stride,
    )


def vor(
    dst: Tensor,
    src1: Tensor,
    src2: Tensor,
    repeat: Union[int, Var, None] = None,
    dst_blk_stride: Union[int, Var, None] = None,
    src1_blk_stride: Union[int, Var, None] = None,
    src2_blk_stride: Union[int, Var, None] = None,
    dst_rep_stride: Union[int, Var, None] = None,
    src1_rep_stride: Union[int, Var, None] = None,
    src2_rep_stride: Union[int, Var, None] = None,
) -> None:
    _validate_binary_tensors(dst, src1, src2, (Datatype.int,))

    if repeat is None:
        repeat = infer_repeat(dst)

    dst_blk_stride, dst_rep_stride = resolve_strides(dst, dst_blk_stride, dst_rep_stride)
    src1_blk_stride, src1_rep_stride = resolve_strides(src1, src1_blk_stride, src1_rep_stride)
    src2_blk_stride, src2_rep_stride = resolve_strides(src2, src2_blk_stride, src2_rep_stride)

    validate_var_or_int(repeat, "repeat")
    validate_var_or_int(dst_blk_stride, "dst_blk_stride")
    validate_var_or_int(src1_blk_stride, "src1_blk_stride")
    validate_var_or_int(src2_blk_stride, "src2_blk_stride")
    validate_var_or_int(dst_rep_stride, "dst_rep_stride")
    validate_var_or_int(src1_rep_stride, "src1_rep_stride")
    validate_var_or_int(src2_rep_stride, "src2_rep_stride")

    _emit_binary_inst(
        "OR",
        dst,
        src1,
        src2,
        repeat,
        dst_blk_stride,
        src1_blk_stride,
        src2_blk_stride,
        dst_rep_stride,
        src1_rep_stride,
        src2_rep_stride,
    )


def muladddst(
    dst: Tensor,
    src1: Tensor,
    src2: Tensor,
    repeat: Union[int, Var, None] = None,
    dst_blk_stride: Union[int, Var, None] = None,
    src1_blk_stride: Union[int, Var, None] = None,
    src2_blk_stride: Union[int, Var, None] = None,
    dst_rep_stride: Union[int, Var, None] = None,
    src1_rep_stride: Union[int, Var, None] = None,
    src2_rep_stride: Union[int, Var, None] = None,
) -> None:
    _validate_binary_tensors(dst, src1, src2, (Datatype.float, Datatype.half, Datatype.int))

    if repeat is None:
        repeat = infer_repeat(dst)

    dst_blk_stride, dst_rep_stride = resolve_strides(dst, dst_blk_stride, dst_rep_stride)
    src1_blk_stride, src1_rep_stride = resolve_strides(src1, src1_blk_stride, src1_rep_stride)
    src2_blk_stride, src2_rep_stride = resolve_strides(src2, src2_blk_stride, src2_rep_stride)

    validate_var_or_int(repeat, "repeat")
    validate_var_or_int(dst_blk_stride, "dst_blk_stride")
    validate_var_or_int(src1_blk_stride, "src1_blk_stride")
    validate_var_or_int(src2_blk_stride, "src2_blk_stride")
    validate_var_or_int(dst_rep_stride, "dst_rep_stride")
    validate_var_or_int(src1_rep_stride, "src1_rep_stride")
    validate_var_or_int(src2_rep_stride, "src2_rep_stride")

    _emit_binary_inst(
        "MULADDDST",
        dst,
        src1,
        src2,
        repeat,
        dst_blk_stride,
        src1_blk_stride,
        src2_blk_stride,
        dst_rep_stride,
        src1_rep_stride,
        src2_rep_stride,
    )
