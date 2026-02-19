from typing import Union

from ...utils.Tensor import Tensor
from ...utils.var import Var
from ...utils.datatype import Datatype
from ...utils.positions import Position
from ...utils.instruction import Instruction
from ... import globvars
from .vecutils import validate_var_or_int


def _infer_sort32_repeat(src: Tensor) -> Union[int, Var]:
    span = src.span if hasattr(src, "span") else src.shape
    dim0 = span[0]
    dim1 = span[1]
    validate_var_or_int(dim0, "shape[0]")
    validate_var_or_int(dim1, "shape[1]")
    count = dim0 * dim1
    if isinstance(count, Var):
        return count // 32
    if not isinstance(count, int):
        raise TypeError(f"repeat cannot be inferred from shape, current count type: {type(count)}")
    return count // 32


def sort32(dst: Tensor, src: Tensor, idx: Tensor, repeat: Union[Var, int, None] = None) -> None:
    if not isinstance(dst, Tensor):
        raise TypeError(f"dst must be Tensor type, current type: {type(dst)}")
    if not isinstance(src, Tensor):
        raise TypeError(f"src must be Tensor type, current type: {type(src)}")
    if not isinstance(idx, Tensor):
        raise TypeError(f"idx must be Tensor type, current type: {type(idx)}")
    if src.position is not Position.UB:
        raise ValueError(f"src must be at UB position, current position: {src.position}")
    if dst.position is not Position.UB:
        raise ValueError(f"dst must be at UB position, current position: {dst.position}")
    if idx.position is not Position.UB:
        raise ValueError(f"idx must be at UB position, current position: {idx.position}")
    if src.dtype != dst.dtype:
        raise ValueError("src/dst data types must match")
    if dst.dtype not in (Datatype.float, Datatype.half):
        raise ValueError(f"does not support data type: {dst.dtype}")
    if idx.dtype is not Datatype.uint32:
        raise ValueError(f"idx must be uint32 type, current type: {idx.dtype}") 
    if repeat is None:
        repeat = _infer_sort32_repeat(src)

    validate_var_or_int(repeat, "repeat")

    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction("sort32", dst=dst, src=src, idx=idx, repeat=repeat)
        )


def mergesort4(
    dst: Tensor,
    src: Tensor,
    length_per_blk: int = 32,
    repeat: Union[Var, int] = 1,
) -> None:
    if not isinstance(dst, Tensor):
        raise TypeError(f"dst must be Tensor type, current type: {type(dst)}")
    if not isinstance(src, Tensor):
        raise TypeError(f"src must be Tensor type, current type: {type(src)}")
    if src.position is not Position.UB:
        raise ValueError(f"src must be at UB position, current position: {src.position}")
    if dst.position is not Position.UB:
        raise ValueError(f"dst must be at UB position, current position: {dst.position}")
    if src.dtype != dst.dtype:
        raise ValueError("src/dst data types must match")
    if dst.dtype not in (Datatype.float, Datatype.half):
        raise ValueError(f"does not support data type: {dst.dtype}")
    if not isinstance(length_per_blk, int):
        raise TypeError(f"length_per_blk must be inttype, current type: {type(length_per_blk)}")
    validate_var_or_int(repeat, "repeat")

    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction(
                "mergesort4",
                dst=dst,
                src=src,
                length_per_blk=length_per_blk,
                repeat=repeat,
            )
        )


def mergesort_2seq(
    dst: Tensor,
    src1: Tensor,
    src2: Tensor,
    size1: Union[int, Var],
    size2: Union[int, Var],
) -> None:
    if not isinstance(dst, Tensor):
        raise TypeError(f"dst must be Tensor type, current type: {type(dst)}")
    if not isinstance(src1, Tensor):
        raise TypeError(f"src1 must be Tensor type, current type: {type(src1)}")
    if not isinstance(src2, Tensor):
        raise TypeError(f"src2 must be Tensor type, current type: {type(src2)}")
    if src1.position is not Position.UB:
        raise ValueError(f"src1 must be at UB position, current position: {src1.position}")
    if src2.position is not Position.UB:
        raise ValueError(f"src2 must be at UB position, current position: {src2.position}")
    if dst.position is not Position.UB:
        raise ValueError(f"dst must be at UB position, current position: {dst.position}")
    if src1.dtype != dst.dtype or src2.dtype != dst.dtype:
        raise ValueError("src1/src2/dst data types must match")
    if dst.dtype not in (Datatype.float, Datatype.half):
        raise ValueError(f"does not support data type: {dst.dtype}")
    validate_var_or_int(size1, "size1")
    validate_var_or_int(size2, "size2")

    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction(
                "mergesort_2seq",
                dst=dst,
                src1=src1,
                src2=src2,
                size1=size1,
                size2=size2,
            )
        )
