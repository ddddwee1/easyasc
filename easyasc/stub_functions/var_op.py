from typing import Optional, Union

from ..utils.var import Var
from ..utils.datatype import Datatype
from ..utils.instruction import Instruction
from .. import globvars


def _value_of(value: Union[Var, int, float]) -> Optional[Union[int, float]]:
    if isinstance(value, Var):
        if isinstance(value.value, (int, float)):
            return value.value
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def CeilDiv(a: Union[Var, int], b: Union[Var, int], *, name: str = "") -> Var:
    if not isinstance(a, (Var, int)):
        raise TypeError(f"a必须是Var或int类型，当前类型: {type(a)}")
    if not isinstance(b, (Var, int)):
        raise TypeError(f"b必须是Var或int类型，当前类型: {type(b)}")
    if not isinstance(name, str):
        raise TypeError(f"name必须是str类型，当前类型: {type(name)}")

    out = Var(name=name)
    a_val = _value_of(a)
    b_val = _value_of(b)
    if a_val is not None and b_val is not None and b_val != 0:
        out.value = (a_val + b_val - 1) // b_val
    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction("CeilDiv", a=a, b=b, out=out)
        )
    return out


def GetCubeNum(*, name: str = "") -> Var:
    if not isinstance(name, str):
        raise TypeError(f"name必须是str类型，当前类型: {type(name)}")
    out = Var(name=name)
    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction("GetCubeNum", out=out)
        )
    return out


def GetCubeIdx(*, name: str = "") -> Var:
    if not isinstance(name, str):
        raise TypeError(f"name必须是str类型，当前类型: {type(name)}")
    out = Var(name=name)
    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction("GetCubeIdx", out=out)
        )
    return out


def var_mul(a: Union[Var, int, float], b: Union[Var, int, float], *, name: str = "") -> Var:
    if not isinstance(a, (Var, int, float)):
        raise TypeError(f"a必须是Var或数值类型，当前类型: {type(a)}")
    if not isinstance(b, (Var, int, float)):
        raise TypeError(f"b必须是Var或数值类型，当前类型: {type(b)}")
    if not isinstance(name, str):
        raise TypeError(f"name必须是str类型，当前类型: {type(name)}")

    dtype = None
    if isinstance(a, float) or isinstance(b, float):
        dtype = Datatype.float
    elif isinstance(a, Var) and a.dtype is Datatype.float:
        dtype = Datatype.float
    elif isinstance(b, Var) and b.dtype is Datatype.float:
        dtype = Datatype.float
    elif isinstance(a, int) or isinstance(b, int):
        dtype = Datatype.int
    elif isinstance(a, Var) and a.dtype is Datatype.int:
        dtype = Datatype.int
    elif isinstance(b, Var) and b.dtype is Datatype.int:
        dtype = Datatype.int

    out = Var(dtype=dtype, name=name)
    a_val = _value_of(a)
    b_val = _value_of(b)
    if a_val is not None and b_val is not None:
        out.value = a_val * b_val
    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction("var_mul", a=a, b=b, out=out)
        )
    return out


def var_add(a: Union[Var, int, float], b: Union[Var, int, float], *, name: str = "") -> Var:
    if not isinstance(a, (Var, int, float)):
        raise TypeError(f"a必须是Var或数值类型，当前类型: {type(a)}")
    if not isinstance(b, (Var, int, float)):
        raise TypeError(f"b必须是Var或数值类型，当前类型: {type(b)}")
    if not isinstance(name, str):
        raise TypeError(f"name必须是str类型，当前类型: {type(name)}")

    dtype = None
    if isinstance(a, float) or isinstance(b, float):
        dtype = Datatype.float
    elif isinstance(a, Var) and a.dtype is Datatype.float:
        dtype = Datatype.float
    elif isinstance(b, Var) and b.dtype is Datatype.float:
        dtype = Datatype.float
    elif isinstance(a, int) or isinstance(b, int):
        dtype = Datatype.int
    elif isinstance(a, Var) and a.dtype is Datatype.int:
        dtype = Datatype.int
    elif isinstance(b, Var) and b.dtype is Datatype.int:
        dtype = Datatype.int

    out = Var(dtype=dtype, name=name)
    a_val = _value_of(a)
    b_val = _value_of(b)
    if a_val is not None and b_val is not None:
        out.value = a_val + b_val
    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction("var_add", a=a, b=b, out=out)
        )
    return out


def var_sub(a: Union[Var, int, float], b: Union[Var, int, float], *, name: str = "") -> Var:
    if not isinstance(a, (Var, int, float)):
        raise TypeError(f"a必须是Var或数值类型，当前类型: {type(a)}")
    if not isinstance(b, (Var, int, float)):
        raise TypeError(f"b必须是Var或数值类型，当前类型: {type(b)}")
    if not isinstance(name, str):
        raise TypeError(f"name必须是str类型，当前类型: {type(name)}")

    dtype = None
    if isinstance(a, float) or isinstance(b, float):
        dtype = Datatype.float
    elif isinstance(a, Var) and a.dtype is Datatype.float:
        dtype = Datatype.float
    elif isinstance(b, Var) and b.dtype is Datatype.float:
        dtype = Datatype.float
    elif isinstance(a, int) or isinstance(b, int):
        dtype = Datatype.int
    elif isinstance(a, Var) and a.dtype is Datatype.int:
        dtype = Datatype.int
    elif isinstance(b, Var) and b.dtype is Datatype.int:
        dtype = Datatype.int

    out = Var(dtype=dtype, name=name)
    a_val = _value_of(a)
    b_val = _value_of(b)
    if a_val is not None and b_val is not None:
        out.value = a_val - b_val
    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction("var_sub", a=a, b=b, out=out)
        )
    return out


def var_div(a: Union[Var, int, float], b: Union[Var, int, float], *, name: str = "") -> Var:
    if not isinstance(a, (Var, int, float)):
        raise TypeError(f"a必须是Var或数值类型，当前类型: {type(a)}")
    if not isinstance(b, (Var, int, float)):
        raise TypeError(f"b必须是Var或数值类型，当前类型: {type(b)}")
    if not isinstance(name, str):
        raise TypeError(f"name必须是str类型，当前类型: {type(name)}")

    def _dtype_of(value, label: str):
        if isinstance(value, Var):
            if value.dtype is None:
                raise TypeError(f"{label}的dtype为None，无法推断")
            return value.dtype
        if isinstance(value, float):
            return Datatype.float
        if isinstance(value, int):
            return Datatype.int
        raise TypeError(f"{label}必须是Var或数值类型，当前类型: {type(value)}")

    dtype_a = _dtype_of(a, "a")
    dtype_b = _dtype_of(b, "b")
    if dtype_a is not dtype_b:
        raise TypeError(f"a与b的数据类型必须一致，当前为{dtype_a}与{dtype_b}")

    out = Var(dtype=dtype_a, name=name)
    a_val = _value_of(a)
    b_val = _value_of(b)
    if a_val is not None and b_val is not None and b_val != 0:
        if dtype_a is Datatype.int:
            out.value = a_val // b_val
        else:
            out.value = a_val / b_val
    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction("var_div", a=a, b=b, out=out)
        )
    return out


def Min(a: Union[Var, int, float], b: Union[Var, int, float], *, name: str = "") -> Var:
    if not isinstance(a, (Var, int, float)):
        raise TypeError(f"a必须是Var或数值类型，当前类型: {type(a)}")
    if not isinstance(b, (Var, int, float)):
        raise TypeError(f"b必须是Var或数值类型，当前类型: {type(b)}")
    if not isinstance(name, str):
        raise TypeError(f"name必须是str类型，当前类型: {type(name)}")

    dtype = None
    if isinstance(a, float) or isinstance(b, float):
        dtype = Datatype.float
    elif isinstance(a, Var) and a.dtype is Datatype.float:
        dtype = Datatype.float
    elif isinstance(b, Var) and b.dtype is Datatype.float:
        dtype = Datatype.float
    elif isinstance(a, int) or isinstance(b, int):
        dtype = Datatype.int
    elif isinstance(a, Var) and a.dtype is Datatype.int:
        dtype = Datatype.int
    elif isinstance(b, Var) and b.dtype is Datatype.int:
        dtype = Datatype.int

    out = Var(dtype=dtype, name=name)
    a_val = _value_of(a)
    b_val = _value_of(b)
    if a_val is not None and b_val is not None:
        out.value = min(a_val, b_val)
    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction("Min", a=a, b=b, out=out)
        )
    return out


def Max(a: Union[Var, int, float], b: Union[Var, int, float], *, name: str = "") -> Var:
    if not isinstance(a, (Var, int, float)):
        raise TypeError(f"a必须是Var或数值类型，当前类型: {type(a)}")
    if not isinstance(b, (Var, int, float)):
        raise TypeError(f"b必须是Var或数值类型，当前类型: {type(b)}")
    if not isinstance(name, str):
        raise TypeError(f"name必须是str类型，当前类型: {type(name)}")

    dtype = None
    if isinstance(a, float) or isinstance(b, float):
        dtype = Datatype.float
    elif isinstance(a, Var) and a.dtype is Datatype.float:
        dtype = Datatype.float
    elif isinstance(b, Var) and b.dtype is Datatype.float:
        dtype = Datatype.float
    elif isinstance(a, int) or isinstance(b, int):
        dtype = Datatype.int
    elif isinstance(a, Var) and a.dtype is Datatype.int:
        dtype = Datatype.int
    elif isinstance(b, Var) and b.dtype is Datatype.int:
        dtype = Datatype.int

    out = Var(dtype=dtype, name=name)
    a_val = _value_of(a)
    b_val = _value_of(b)
    if a_val is not None and b_val is not None:
        out.value = max(a_val, b_val)
    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction("Max", a=a, b=b, out=out)
        )
    return out
