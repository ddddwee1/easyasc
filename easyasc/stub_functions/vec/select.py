from typing import Union

from ...utils.Tensor import Tensor
from ...utils.var import Var
from ...utils.datatype import Datatype
from ...utils.positions import Position
from ...utils.instruction import Instruction
from ...utils.selectmode import SelectMode, SelectModeType
from ... import globvars
from .vecutils import infer_repeat, resolve_strides, validate_var_or_int, validate_scalar


def _validate_selmask(selmask) -> None:
    if isinstance(selmask, Tensor):
        if selmask.dtype not in (Datatype.uint8,):
            raise ValueError(f"selmask data type is not supported, current type: {selmask.dtype}")
    elif isinstance(selmask, Var):
        if selmask.dtype is None:
            raise TypeError("selmask dtype is None, cannot infer")
        if selmask.dtype not in (Datatype.uint8, Datatype.uint16, Datatype.uint32, Datatype.uint64):
            raise ValueError(f"selmask data type is not supported, current type: {selmask.dtype}")
    else:
        raise TypeError(f"selmask must be Tensor or Var, current type: {type(selmask)}")

def select(
    dst: Tensor,
    selmask: Union[Tensor, Var],
    src1: Tensor,
    src2: Union[Tensor, Var, int, float],
    repeat: Union[int, Var, None] = None,
    dst_blk_stride: Union[int, Var, None] = None,
    src1_blk_stride: Union[int, Var, None] = None,
    src2_blk_stride: Union[int, Var, None] = None,
    dst_rep_stride: Union[int, Var, None] = None,
    src1_rep_stride: Union[int, Var, None] = None,
    src2_rep_stride: Union[int, Var, None] = None,
) -> None:
    if not isinstance(dst, Tensor):
        raise TypeError(f"dst must be Tensor type, current type: {type(dst)}")
    if not isinstance(src1, Tensor):
        raise TypeError(f"src1 must be Tensor type, current type: {type(src1)}")
    if dst.position is not Position.UB:
        raise ValueError(f"dst must be at UB position, current position: {dst.position}")
    if src1.position is not Position.UB:
        raise ValueError(f"src1 must be at UB position, current position: {src1.position}")
    if dst.dtype != src1.dtype:
        raise ValueError("dst/src1 data types must match")
    if dst.dtype not in (Datatype.float, Datatype.half, Datatype.int16, Datatype.int):
        raise ValueError(f"dst data type is not supported, current type: {dst.dtype}")
    _validate_selmask(selmask)

    if isinstance(src2, Tensor):
        if src2.position is not Position.UB:
            raise ValueError(f"src2 must be at UB position, current position: {src2.position}")
        if src2.dtype != dst.dtype:
            raise ValueError("src2 and dst data types must match")
        mode = SelectMode.TENSOR_TENSOR
    else:
        validate_scalar(src2, "src2")
        mode = SelectMode.TENSOR_SCALAR

    if repeat is None:
        repeat = infer_repeat(dst)

    dst_blk_stride, dst_rep_stride = resolve_strides(dst, dst_blk_stride, dst_rep_stride)
    src1_blk_stride, src1_rep_stride = resolve_strides(src1, src1_blk_stride, src1_rep_stride)
    if isinstance(src2, Tensor):
        src2_blk_stride, src2_rep_stride = resolve_strides(src2, src2_blk_stride, src2_rep_stride)
    else:
        if src2_blk_stride is None:
            src2_blk_stride = 1
        if src2_rep_stride is None:
            src2_rep_stride = 8

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
                "select",
                dst=dst,
                mode=mode,
                selmask=selmask,
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
