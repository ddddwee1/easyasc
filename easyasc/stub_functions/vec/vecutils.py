from typing import Union, Tuple

from ...utils.Tensor import Tensor
from ...utils.var import Var
from ..var_op import CeilDiv


def validate_var_or_int(value: object, label: str) -> None:
    if not isinstance(value, (Var, int)):
        raise TypeError(f"{label} must be Var or int type, current type: {type(value)}") 

def validate_scalar(value: object, label: str) -> None:
    if not isinstance(value, (Var, int, float)):
        raise TypeError(f"{label} must be Var or numeric value type, current type: {type(value)}") 

def infer_repeat(tensor: Tensor) -> Union[int, Var]:
    span = tensor.span if hasattr(tensor, "span") else tensor.shape
    dim0 = span[0]
    dim1 = span[1]
    validate_var_or_int(dim0, "shape[0]")
    validate_var_or_int(dim1, "shape[1]")

    count = dim0 * dim1
    denom = 256 // tensor.dtype.size
    if isinstance(count, Var):
        return CeilDiv(count, denom)
    if not isinstance(count, int):
        raise TypeError(f"repeat cannot be inferred from shape, current count type: {type(count)}")
    return count // denom


def infer_repeat_brcb(src: Tensor) -> Union[int, Var]:
    span = src.span if hasattr(src, "span") else src.shape
    dim0 = span[0]
    dim1 = span[1]
    validate_var_or_int(dim0, "shape[0]")
    validate_var_or_int(dim1, "shape[1]")

    count = dim0 * dim1
    if isinstance(count, Var):
        return CeilDiv(count, 8)
    if not isinstance(count, int):
        raise TypeError(f"repeat cannot be inferred from shape, current count type: {type(count)}")
    return count // 8


def infer_rep_stride(shape1: object, c0: int) -> Union[int, Var]:
    if isinstance(shape1, Var):
        return CeilDiv(shape1, c0)
    if isinstance(shape1, int):
        return shape1 // c0
    raise TypeError(f"rep_stride cannot be inferred from shape, current shape[1] type: {type(shape1)}")

def infer_strides(tensor: Tensor) -> Tuple[Union[int, Var], Union[int, Var]]:
    span = tensor.span if hasattr(tensor, "span") else tensor.shape
    shape = tensor.shape
    span0 = span[0]
    span1 = span[1]
    c0 = tensor.dtype.C0

    blk_stride: Union[int, Var] = 1
    rep_stride: Union[int, Var] = 8
    matched = False
    if isinstance(span1, int):
        if span1 == 8 * c0:
            blk_stride = 1
            rep_stride = infer_rep_stride(shape[1], c0)
            matched = True
        elif span1 == c0:
            blk_stride = 0
            rep_stride = infer_rep_stride(shape[1], c0)
            matched = True

    if matched and isinstance(span0, int) and span0 == 1:
        rep_stride = 0

    return blk_stride, rep_stride


def resolve_strides(
    tensor: Tensor,
    blk_stride: Union[int, Var, None],
    rep_stride: Union[int, Var, None],
) -> Tuple[Union[int, Var], Union[int, Var]]:
    if blk_stride is None or rep_stride is None:
        auto_blk, auto_rep = infer_strides(tensor)
        if blk_stride is None:
            blk_stride = auto_blk
        if rep_stride is None:
            rep_stride = auto_rep
    return blk_stride, rep_stride
