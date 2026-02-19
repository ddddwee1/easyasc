from __future__ import annotations

from typing import Optional, Union, cast
from .datatype import DataTypeValue, Datatype
from .. import globvars
from .instruction import Instruction


class Expr:
    def __init__(self, expr: str):
        self.expr = expr

    def __repr__(self):
        return f"Expr({self.expr!r})"

    def __str__(self):
        return self.expr

    def __bool__(self) -> bool:
        raise TypeError("Expr cannot be used in Python boolean context, use If/Elif/Else instead")
        return False

    def __and__(self, other):
        return Expr(f"({self}) && ({other})")

    def __or__(self, other):
        return Expr(f"({self}) || ({other})")

    def __invert__(self):
        return Expr(f"!({self})")


def _expr_operand(value) -> str:
    if isinstance(value, Var):
        return value.name
    if isinstance(value, Expr):
        return str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


class Var:
    """Variable class."""
    def __init__(
        self,
        value: Union[int, float, 'Var', None] = 0,
        dtype: Optional[DataTypeValue] = None,
        name: str = "",
    ):
        """
        Initialize a variable.

        Args:
            value: Optional value, defaults to None.
            dtype: Data type, must be DataTypeValue.
            name: Name, defaults to empty string.
        """
        value_var: Optional[Var] = value if isinstance(value, Var) else None
        value_is_var = value_var is not None

        if dtype is None:
            if isinstance(value, int):
                dtype = Datatype.int
            elif isinstance(value, float):
                dtype = Datatype.float
            elif value_var is not None:
                dtype = value_var.dtype
            elif value is not None:
                raise TypeError(f"Cannot infer dtype from value type: {type(value)}")

        if dtype is not None and not isinstance(dtype, DataTypeValue):
            raise TypeError(f"dtype must be DataTypeValue or None, got: {type(dtype)}")
        
        if not isinstance(name, str):
            raise TypeError(f"name must be str, got: {type(name)}")
        
        if value is not None and not isinstance(value, (int, float, Var)):
            raise TypeError(f"value must be int, float, or Var, got: {type(value)}")
        
        if value_var is not None:
            idx = value_var.idx
        else:
            idx = globvars.tmp_idx
            globvars.tmp_idx += 1
            if name == "":
                name = f"_tmp_var_{idx}"

        self.dtype: Optional[DataTypeValue] = dtype
        self.name: str = name
        if value_var is not None:
            value_var.name = self.name
        if value_var is not None:
            self.value: Union[int, float, None] = value_var.value
        else:
            self.value = cast(Union[int, float, None], value)
        self.idx: int = idx

        if not value_is_var and not isinstance(value, Var):
            if globvars.active_micro is not None:
                globvars.active_micro.instructions.append(
                    Instruction("create_var", val=self)
                )
            elif globvars.active_kernel is not None:
                globvars.active_kernel.instructions.append(
                    Instruction("create_var", val=self)
                )

    def __repr__(self):
        return (
            f"Var(name={self.name!r}, dtype={self.dtype!r}, value={self.value!r}, "
            f"idx={self.idx!r})"
        )

    def __mul__(self, other):
        from ..stub_functions.var_op import var_mul
        return var_mul(self, other)

    def __rmul__(self, other):
        from ..stub_functions.var_op import var_mul
        return var_mul(other, self)

    def __add__(self, other):
        from ..stub_functions.var_op import var_add
        return var_add(self, other)

    def __radd__(self, other):
        from ..stub_functions.var_op import var_add
        return var_add(other, self)

    def __iadd__(self, other):
        if not isinstance(other, (Var, int, float)):
            raise TypeError(f"other must be Var or numeric type, got: {type(other)}")

        dtype = None
        if isinstance(other, float) or (isinstance(other, Var) and other.dtype is Datatype.float):
            dtype = Datatype.float
        elif self.dtype is Datatype.float:
            dtype = Datatype.float
        elif isinstance(other, int) or (isinstance(other, Var) and other.dtype is Datatype.int):
            dtype = Datatype.int
        elif self.dtype is Datatype.int:
            dtype = Datatype.int

        if dtype is not None:
            self.dtype = dtype

        target = globvars.active_micro if globvars.active_micro is not None else globvars.active_kernel
        if target is not None:
            target.instructions.append(
                Instruction("var_add", a=self, b=other, out=self)
            )
        return self

    def __sub__(self, other):
        from ..stub_functions.var_op import var_sub
        return var_sub(self, other)

    def __rsub__(self, other):
        from ..stub_functions.var_op import var_sub
        return var_sub(other, self)

    def __truediv__(self, other):
        from ..stub_functions.var_op import var_div
        return var_div(self, other)

    def __rtruediv__(self, other):
        from ..stub_functions.var_op import var_div
        return var_div(other, self)

    def __floordiv__(self, other):
        from ..stub_functions.var_op import var_div
        return var_div(self, other)

    def __rfloordiv__(self, other):
        from ..stub_functions.var_op import var_div
        return var_div(other, self)

    def _cmp(self, other, op: str):
        return Expr(f"{_expr_operand(self)} {op} {_expr_operand(other)}")

    def __eq__(self, other):
        return self._cmp(other, "==")

    def __ne__(self, other):
        return self._cmp(other, "!=")

    def __lt__(self, other):
        return self._cmp(other, "<")

    def __le__(self, other):
        return self._cmp(other, "<=")

    def __gt__(self, other):
        return self._cmp(other, ">")

    def __ge__(self, other):
        return self._cmp(other, ">=")
