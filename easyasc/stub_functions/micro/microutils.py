from typing import Optional, Union

from ... import globvars
from ...utils.datatype import DataTypeValue
from ...utils.reg import MaskReg
from ...utils.var import Var


def require_micro():
    micro = globvars.active_micro
    if micro is None:
        raise RuntimeError("micro函数只能在MicroModule中调用")
    return micro


def ensure_mask(mask: Optional[MaskReg], dtype: DataTypeValue, micro) -> MaskReg:
    if mask is None:
        mask = micro.get_mask(dtype)
    if not isinstance(mask, MaskReg):
        raise TypeError(f"mask必须是MaskReg类型，当前类型: {type(mask)}")
    return mask


def format_scalar(value: Union[Var, int, float], dtype: DataTypeValue) -> str:
    if isinstance(value, Var):
        return value.name
    if not isinstance(value, (int, float)):
        raise TypeError(f"value必须是Var或数值类型，当前类型: {type(value)}")
    if not isinstance(dtype, DataTypeValue):
        raise TypeError(f"dtype必须是DataTypeValue类型，当前类型: {type(dtype)}")

    dtype_name = getattr(dtype, "name", None)
    if dtype_name == "float":
        return f"{float(value)}f"
    if dtype_name == "half":
        return f"(half){float(value)}f"
    if dtype_name == "bfloat16_t":
        return f"(bfloat16_t){float(value)}f"
    if isinstance(value, float):
        value = int(value)
    return f"{int(value)}"


def dtype_size(dtype: DataTypeValue) -> int:
    if not isinstance(dtype, DataTypeValue):
        raise TypeError(f"dtype必须是DataTypeValue类型，当前类型: {type(dtype)}")
    return dtype.size
