from typing import Union

from .var import Var
from .Tensor import Tensor


class VecOP:
    """
    Lightweight wrapper that records a vector operation until a destination
    tensor is provided via <<=. This keeps syntax sugar like:
        dst <<= src1 + src2
    while still emitting the correct low-level vector instructions.
    """

    def __init__(self, op: str, src1: Tensor, src2: Union[Tensor, int, float, Var, None] = None):
        self.op = op
        self.src1 = src1
        self.src2 = src2

    def emit(self, dst: Tensor) -> None:
        """
        Materialize the recorded op into the concrete vector instruction
        using the provided destination tensor.
        """
        op = self.op

        # Binary vector ops.
        if op in ("add", "sub", "mul", "div", "max", "min", "and", "or"):
            if not isinstance(self.src2, Tensor):
                raise TypeError("src2 must be a Tensor for binary vec ops")
            src2 = self.src2
            from ..stub_functions.vec.binary import add, sub, mul, div, vmax, vmin, vand, vor
            if op == "add":
                add(dst, self.src1, src2)
            elif op == "sub":
                sub(dst, self.src1, src2)
            elif op == "mul":
                mul(dst, self.src1, src2)
            elif op == "div":
                div(dst, self.src1, src2)
            elif op == "max":
                vmax(dst, self.src1, src2)
            elif op == "min":
                vmin(dst, self.src1, src2)
            elif op == "and":
                from ..utils.datatype import Datatype
                if dst.dtype is Datatype.int and self.src1.dtype is Datatype.int:
                    vand(dst, self.src1, src2)
                else:
                    from ..stub_functions.misc import reinterpret
                    src1_int = reinterpret(self.src1, Datatype.int)
                    src2_int = reinterpret(src2, Datatype.int)
                    dst_int = reinterpret(dst, Datatype.int)
                    vand(dst_int, src1_int, src2_int)
            else:
                from ..utils.datatype import Datatype
                if dst.dtype is Datatype.int and self.src1.dtype is Datatype.int:
                    vor(dst, self.src1, src2)
                else:
                    from ..stub_functions.misc import reinterpret
                    src1_int = reinterpret(self.src1, Datatype.int)
                    src2_int = reinterpret(src2, Datatype.int)
                    dst_int = reinterpret(dst, Datatype.int)
                    vor(dst_int, src1_int, src2_int)
            return

        # Vector + scalar or vector * scalar (and scalar min/max).
        if op in ("adds", "muls", "maxs", "mins"):
            from ..stub_functions.vec.unaryscalar import adds, muls, vmaxs, vmins
            if op == "adds":
                adds(dst, self.src1, self.src2)  # type: ignore[arg-type]
            elif op == "muls":
                muls(dst, self.src1, self.src2)  # type: ignore[arg-type]
            elif op == "maxs":
                vmaxs(dst, self.src1, self.src2)  # type: ignore[arg-type]
            else:
                vmins(dst, self.src1, self.src2)  # type: ignore[arg-type]
            return

        # Unary vector ops.
        if op in ("exp", "ln", "abs", "rec", "sqrt", "rsqrt", "vnot", "relu"):
            from ..stub_functions.vec.unary import exp, ln, abs, rec, sqrt, rsqrt, vnot, relu
            if op == "exp":
                exp(dst, self.src1)
            elif op == "ln":
                ln(dst, self.src1)
            elif op == "abs":
                abs(dst, self.src1)
            elif op == "rec":
                rec(dst, self.src1)
            elif op == "sqrt":
                sqrt(dst, self.src1)
            elif op == "rsqrt":
                rsqrt(dst, self.src1)
            elif op == "vnot":
                # vnot only supports int; reinterpret float/half views if needed.
                from ..utils.datatype import Datatype
                if dst.dtype is Datatype.int and self.src1.dtype is Datatype.int:
                    vnot(dst, self.src1)
                else:
                    from ..stub_functions.misc import reinterpret
                    src_int = reinterpret(self.src1, Datatype.int)
                    dst_int = reinterpret(dst, Datatype.int)
                    vnot(dst_int, src_int)
            else:
                relu(dst, self.src1)
            return

        # Group ops.
        if op in ("cmax", "cgmax", "cmin", "cgmin", "cadd", "cgadd", "cpadd"):
            from ..stub_functions.vec.group import (
                cmax,
                cgmax,
                cmin,
                cgmin,
                cadd,
                cgadd,
                cpadd,
            )
            if op == "cmax":
                cmax(dst, self.src1)
            elif op == "cgmax":
                cgmax(dst, self.src1)
            elif op == "cmin":
                cmin(dst, self.src1)
            elif op == "cgmin":
                cgmin(dst, self.src1)
            elif op == "cadd":
                cadd(dst, self.src1)
            elif op == "cgadd":
                cgadd(dst, self.src1)
            else:
                cpadd(dst, self.src1)
            return

        # Cast op.
        if op == "cast":
            from ..stub_functions.vec.cast import cast
            cast(dst, self.src1)
            return

        raise ValueError(f"Unsupported VecOP: {op}")


def maximum(a: Union[Tensor, int, float, Var], b: Union[Tensor, int, float, Var]) -> VecOP:
    """Vector maximum: returns a VecOP to be consumed by <<=."""
    if isinstance(a, Tensor) and isinstance(b, Tensor):
        return VecOP("max", a, b)
    if isinstance(a, Tensor) and isinstance(b, (int, float, Var)):
        return VecOP("maxs", a, b)
    if isinstance(b, Tensor) and isinstance(a, (int, float, Var)):
        return VecOP("maxs", b, a)
    raise TypeError("maximum需要Tensor或标量(含Var)入参")


def minimum(a: Union[Tensor, int, float, Var], b: Union[Tensor, int, float, Var]) -> VecOP:
    """Vector minimum: returns a VecOP to be consumed by <<=."""
    if isinstance(a, Tensor) and isinstance(b, Tensor):
        return VecOP("min", a, b)
    if isinstance(a, Tensor) and isinstance(b, (int, float, Var)):
        return VecOP("mins", a, b)
    if isinstance(b, Tensor) and isinstance(a, (int, float, Var)):
        return VecOP("mins", b, a)
    raise TypeError("minimum需要Tensor或标量(含Var)入参")
