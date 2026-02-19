from typing import Optional, Union

from ... import globvars
from ...utils.datatype import DataTypeValue
from ...utils.reg import MaskReg
from ...utils.var import Var


def require_micro():
    micro = globvars.active_micro
    if micro is None:
        raise RuntimeError("micro function can only be used in MicroModule")
    return micro


def ensure_mask(mask: Optional[MaskReg], dtype: DataTypeValue, micro) -> MaskReg:
    if mask is None:
        mask = micro.get_mask(dtype)
    if not isinstance(mask, MaskReg):
        raise TypeError(f"mask must be MaskReg type, current type: {type(mask)}")
    return mask


def format_scalar(value: Union[Var, int, float], dtype: DataTypeValue) -> str:
    if isinstance(value, Var):
        return value.name
    if not isinstance(value, (int, float)):
        raise TypeError(f"value must be Var or numeric value type, current type: {type(value)}")
    if not isinstance(dtype, DataTypeValue):
        raise TypeError(f"dtype must be DataTypeValue type, current type: {type(dtype)}") 
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
        raise TypeError(f"dtype must be DataTypeValue type, current type: {type(dtype)}")
    return dtype.size
