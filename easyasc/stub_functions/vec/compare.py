from typing import Union

from ...utils.Tensor import Tensor
from ...utils.var import Var
from ...utils.datatype import Datatype
from ...utils.positions import Position
from ...utils.instruction import Instruction
from ...utils.comparemode import CompareModeType
from ... import globvars
from .vecutils import infer_repeat, resolve_strides, validate_var_or_int, validate_scalar


def _validate_compare_tensors(dst: Tensor, src1: Tensor, src2: Tensor) -> None:
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
    if src1.dtype != src2.dtype:
        raise ValueError("src1/src2数据类型必须一致")
    if dst.dtype not in (Datatype.int8, Datatype.uint8):
        raise ValueError(f"dst必须是int8/uint8，当前类型: {dst.dtype}")
    if src1.dtype not in (Datatype.float, Datatype.half, Datatype.int16, Datatype.int):
        raise ValueError(f"src1类型不支持，当前类型: {src1.dtype}")


def compare(
    dst: Tensor,
    src1: Tensor,
    src2: Tensor,
    mode: CompareModeType,
    repeat: Union[int, Var, None] = None,
    dst_blk_stride: Union[int, Var, None] = None,
    src1_blk_stride: Union[int, Var, None] = None,
    src2_blk_stride: Union[int, Var, None] = None,
    dst_rep_stride: Union[int, Var, None] = None,
    src1_rep_stride: Union[int, Var, None] = None,
    src2_rep_stride: Union[int, Var, None] = None,
) -> None:
    _validate_compare_tensors(dst, src1, src2)
    if not isinstance(mode, CompareModeType):
        raise TypeError(f"mode必须是CompareModeType类型，当前类型: {type(mode)}")

    if repeat is None:
        repeat = infer_repeat(src1)

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

    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction(
                "COMPARE",
                dst=dst,
                src1=src1,
                src2=src2,
                mode=mode,
                repeat=repeat,
                dst_blk_stride=dst_blk_stride,
                src1_blk_stride=src1_blk_stride,
                src2_blk_stride=src2_blk_stride,
                dst_rep_stride=dst_rep_stride,
                src1_rep_stride=src1_rep_stride,
                src2_rep_stride=src2_rep_stride,
            )
        )


def compare_scalar(
    dst: Tensor,
    src1: Tensor,
    src2: Union[int, float, Var],
    mode: CompareModeType,
    repeat: Union[int, Var, None] = None,
    dst_blk_stride: Union[int, Var, None] = None,
    src1_blk_stride: Union[int, Var, None] = None,
    dst_rep_stride: Union[int, Var, None] = None,
    src1_rep_stride: Union[int, Var, None] = None,
) -> None:
    if not isinstance(dst, Tensor):
        raise TypeError(f"dst必须是Tensor类型，当前类型: {type(dst)}")
    if not isinstance(src1, Tensor):
        raise TypeError(f"src1必须是Tensor类型，当前类型: {type(src1)}")
    if dst.position is not Position.UB:
        raise ValueError(f"dst必须在UB位置，当前位置: {dst.position}")
    if src1.position is not Position.UB:
        raise ValueError(f"src1必须在UB位置，当前位置: {src1.position}")
    if dst.dtype not in (Datatype.int8, Datatype.uint8):
        raise ValueError(f"dst必须是int8/uint8，当前类型: {dst.dtype}")
    if src1.dtype not in (Datatype.float, Datatype.half, Datatype.int16, Datatype.int):
        raise ValueError(f"src1类型不支持，当前类型: {src1.dtype}")
    if not isinstance(mode, CompareModeType):
        raise TypeError(f"mode必须是CompareModeType类型，当前类型: {type(mode)}")
    validate_scalar(src2, "src2")

    if repeat is None:
        repeat = infer_repeat(src1)

    dst_blk_stride, dst_rep_stride = resolve_strides(dst, dst_blk_stride, dst_rep_stride)
    src1_blk_stride, src1_rep_stride = resolve_strides(src1, src1_blk_stride, src1_rep_stride)

    validate_var_or_int(repeat, "repeat")
    validate_var_or_int(dst_blk_stride, "dst_blk_stride")
    validate_var_or_int(src1_blk_stride, "src1_blk_stride")
    validate_var_or_int(dst_rep_stride, "dst_rep_stride")
    validate_var_or_int(src1_rep_stride, "src1_rep_stride")

    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction(
                "COMPARES",
                dst=dst,
                src1=src1,
                src2=src2,
                mode=mode,
                repeat=repeat,
                dst_blk_stride=dst_blk_stride,
                src1_blk_stride=src1_blk_stride,
                dst_rep_stride=dst_rep_stride,
                src1_rep_stride=src1_rep_stride,
            )
        )


def set_cmpmask(src: Tensor) -> None:
    if not isinstance(src, Tensor):
        raise TypeError(f"src必须是Tensor类型，当前类型: {type(src)}")
    if src.position is not Position.UB:
        raise ValueError(f"src必须在UB位置，当前位置: {src.position}")

    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction("SETCMPMASK", src=src)
        )
