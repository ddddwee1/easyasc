from typing import Union

from ..utils.Tensor import Tensor
from ..utils.var import Var
from ..utils.datatype import DataTypeValue, Datatype
from ..utils.instruction import Instruction
from .. import globvars
from .var_op import var_mul, var_div


def _scale_dim(value: Union[int, Var], num: int, den: int, label: str) -> Union[int, Var]:
    if isinstance(value, Var):
        if value.dtype is not None and value.dtype is not Datatype.int:
            raise TypeError(f"{label}必须是DT.int类型的Var，当前类型: {value.dtype}")
        if num == den:
            return value
        if num % den == 0:
            factor = num // den
            if factor == 1:
                return value
            return var_mul(value, factor)
        if den % num == 0:
            factor = den // num
            return var_div(value, factor)
        return var_div(var_mul(value, num), den)
    if isinstance(value, int):
        if num == den:
            return value
        return (value * num) // den
    raise TypeError(f"{label}必须是int或Var类型，当前类型: {type(value)}")


def reinterpret(src: Tensor, target_dtype: DataTypeValue, name: str = "") -> Tensor:
    if not isinstance(src, Tensor):
        raise TypeError(f"src必须是Tensor类型，当前类型: {type(src)}")
    if not isinstance(target_dtype, DataTypeValue):
        raise TypeError(f"target_dtype必须是DataTypeValue类型，当前类型: {type(target_dtype)}")
    if not isinstance(name, str):
        raise TypeError(f"name必须是str类型，当前类型: {type(name)}")

    src_c0 = src.dtype.C0
    dst_c0 = target_dtype.C0

    shape = list(src.shape)
    span = list(src.span)
    shape1 = shape[1]
    span1 = span[1]
    if not isinstance(shape1, (int, Var)):
        raise TypeError(f"shape[1]必须是int或Var类型，当前类型: {type(shape1)}")
    if not isinstance(span1, (int, Var)):
        raise TypeError(f"span[1]必须是int或Var类型，当前类型: {type(span1)}")

    shape[1] = _scale_dim(shape1, dst_c0, src_c0, "shape[1]")
    span[1] = _scale_dim(span1, dst_c0, src_c0, "span[1]")

    idx = globvars.tmp_idx
    globvars.tmp_idx += 1
    if name == "":
        name = f"_tmp_tensor_{idx}"

    out = object.__new__(Tensor)
    out.dtype = target_dtype
    out.shape = shape
    out.name = name
    out.position = src.position
    out.idx = idx
    out.offset = list(src.offset)
    out.span = span
    out.step = list(src.step)
    out.source_buf = src.source_buf
    out.source_index = src.source_index
    out.is_transpose = src.is_transpose

    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction("REINTERPRET", dst=out, src=src)
        )
    return out
