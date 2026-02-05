from typing import Optional, Union

from ..flowcontrol import If, Else, range as asc_range
from ..stub_functions import Min, mmad, reinterpret
from ..utils.Tensor import Tensor, DBuff
from ..utils.datatype import Datatype as DT
from ..utils.positions import Position
from ..utils.var import Var
from .. import globvars


def _is_transpose(tensor: Tensor) -> bool:
    return bool(getattr(tensor, "is_transpose", getattr(tensor, "is_trans", False)))


def _maybe_int4(tensor: Tensor) -> Tensor:
    dt_int4 = getattr(DT, "int4", None)
    if dt_int4 is None:
        return tensor
    if tensor.dtype in (DT.int, DT.int16):
        return reinterpret(tensor, dt_int4)
    return tensor


def _ensure_var_or_int(value: object, label: str) -> None:
    if not isinstance(value, (Var, int)):
        raise TypeError(f"{label} must be Var or int, got: {type(value)}")


def _prepare_l0_tensors(
    l0a: DBuff,
    l0b: DBuff,
    l1a: Tensor,
    l1b: Tensor,
    l0acnt: Var,
    l0bcnt: Var,
) -> tuple[Tensor, Tensor]:
    if l0a.dtype != l1a.dtype:
        l0a_tmp = reinterpret(l0a[l0acnt], l1a.dtype)
    else:
        l0a_tmp = l0a[l0acnt]
    if l0b.dtype != l1b.dtype:
        l0b_tmp = reinterpret(l0b[l0bcnt], l1b.dtype)
    else:
        l0b_tmp = l0b[l0bcnt]
    return l0a_tmp, l0b_tmp


def matmul_splitn(
    dst: Tensor,
    l1a: Tensor,
    l1b: Tensor,
    l0a: DBuff,
    l0b: DBuff,
    l0acnt: Var,
    l0bcnt: Var,
    split_n: Union[int, Var],
    is_init: Union[bool, int, Var],
) -> None:
    n = l1b.shape[1] if _is_transpose(l1b) else l1b.shape[0]
    for _subn in asc_range(0, n, split_n, name="_subn"):
        valid_n = Min(n - _subn, split_n)
        l0a_tmp, l0b_tmp = _prepare_l0_tensors(l0a, l0b, l1a, l1b, l0acnt, l0bcnt)

        with If(_subn == 0):
            l0a_tmp <<= l1a
        if _is_transpose(l1b):
            l0b_tmp <<= l1b[:, _subn : _subn + valid_n]
        else:
            l0b_tmp <<= l1b[_subn : _subn + valid_n, :]

        l0a_tmp = _maybe_int4(l0a_tmp)
        l0b_tmp = _maybe_int4(l0b_tmp)

        sub_dst = dst[:, _subn : _subn + valid_n]
        if isinstance(is_init, (bool, int)):
            mmad(sub_dst, l0a_tmp, l0b_tmp, is_init=bool(is_init))
        else:
            with If(is_init):
                mmad(sub_dst, l0a_tmp, l0b_tmp, is_init=True)
            with Else():
                mmad(sub_dst, l0a_tmp, l0b_tmp, is_init=False)

        l0bcnt += 1
    l0acnt += 1


def matmul_splitk(
    dst: Tensor,
    l1a: Tensor,
    l1b: Tensor,
    l0a: DBuff,
    l0b: DBuff,
    l0acnt: Var,
    l0bcnt: Var,
    split_k: Union[int, Var],
    is_init: Union[bool, int, Var],
) -> None:
    k = l1b.shape[0] if _is_transpose(l1b) else l1b.shape[1]
    for _subk in asc_range(0, k, split_k, name="_subk"):
        valid_k = Min(k - _subk, split_k)
        l0a_tmp, l0b_tmp = _prepare_l0_tensors(l0a, l0b, l1a, l1b, l0acnt, l0bcnt)

        if _is_transpose(l1a):
            l0a_tmp <<= l1a[_subk : _subk + valid_k, :]
        else:
            l0a_tmp <<= l1a[:, _subk : _subk + valid_k]

        if _is_transpose(l1b):
            l0b_tmp <<= l1b[_subk : _subk + valid_k, :]
        else:
            l0b_tmp <<= l1b[:, _subk : _subk + valid_k]

        l0a_tmp = _maybe_int4(l0a_tmp)
        l0b_tmp = _maybe_int4(l0b_tmp)

        if isinstance(is_init, (bool, int)):
            if bool(is_init):
                with If(_subk == 0):
                    mmad(dst, l0a_tmp, l0b_tmp, is_init=True)
                with Else():
                    mmad(dst, l0a_tmp, l0b_tmp, is_init=False)
            else:
                mmad(dst, l0a_tmp, l0b_tmp, is_init=False)
        else:
            with If((_subk == 0) & is_init):
                mmad(dst, l0a_tmp, l0b_tmp, is_init=True)
            with Else():
                mmad(dst, l0a_tmp, l0b_tmp, is_init=False)

        l0acnt += 1
        l0bcnt += 1


def matmul_nosplit(
    dst: Tensor,
    l1a: Tensor,
    l1b: Tensor,
    l0a: DBuff,
    l0b: DBuff,
    l0acnt: Var,
    l0bcnt: Var,
    is_init: Union[bool, int, Var],
) -> None:
    l0a_tmp, l0b_tmp = _prepare_l0_tensors(l0a, l0b, l1a, l1b, l0acnt, l0bcnt)

    l0a_tmp <<= l1a
    l0b_tmp <<= l1b

    l0a_tmp = _maybe_int4(l0a_tmp)
    l0b_tmp = _maybe_int4(l0b_tmp)

    if isinstance(is_init, (bool, int)):
        mmad(dst, l0a_tmp, l0b_tmp, is_init=bool(is_init))
    else:
        with If(is_init):
            mmad(dst, l0a_tmp, l0b_tmp, is_init=True)
        with Else():
            mmad(dst, l0a_tmp, l0b_tmp, is_init=False)

    l0acnt += 1
    l0bcnt += 1


def matmul(
    dst: Tensor,
    l1a: Tensor,
    l1b: Tensor,
    splitn: Union[int, Var, None] = None,
    splitk: Union[int, Var, None] = None,
    is_init: Union[bool, int, Var] = True,
) -> None:
    if not isinstance(dst, Tensor):
        raise TypeError(f"dst must be Tensor, got: {type(dst)}")
    if dst.position is not Position.L0C:
        raise ValueError(f"dst must be L0C, got: {dst.position}")
    if not isinstance(l1a, Tensor):
        raise TypeError(f"l1a must be Tensor, got: {type(l1a)}")
    if l1a.position is not Position.L1:
        raise ValueError(f"l1a must be L1, got: {l1a.position}")
    if not isinstance(l1b, Tensor):
        raise TypeError(f"l1b must be Tensor, got: {type(l1b)}")
    if l1b.position is not Position.L1:
        raise ValueError(f"l1b must be L1, got: {l1b.position}")
    if splitn is not None:
        _ensure_var_or_int(splitn, "splitn")
    if splitk is not None:
        _ensure_var_or_int(splitk, "splitk")
    if splitn is not None and splitk is not None:
        raise ValueError("splitn and splitk cannot both be set")

    if globvars.active_kernel is None:
        raise RuntimeError("matmul must be called inside a kernel")
    l0acnt = getattr(globvars.active_kernel, "_l0acnt", None)
    l0bcnt = getattr(globvars.active_kernel, "_l0bcnt", None)
    l0a = getattr(globvars.active_kernel, "_l0a", None)
    l0b = getattr(globvars.active_kernel, "_l0b", None)
    if l0acnt is None or l0bcnt is None:
        raise RuntimeError("active_kernel missing _l0acnt/_l0bcnt")
    if l0a is None or l0b is None:
        raise RuntimeError("active_kernel missing _l0a/_l0b")
    if not isinstance(l0a, DBuff) or l0a.position is not Position.L0A:
        raise TypeError(f"_l0a must be L0A DBuff, got: {type(l0a)} {getattr(l0a, 'position', None)}")
    if not isinstance(l0b, DBuff) or l0b.position is not Position.L0B:
        raise TypeError(f"_l0b must be L0B DBuff, got: {type(l0b)} {getattr(l0b, 'position', None)}")

    if splitn is not None:
        matmul_splitn(dst, l1a, l1b, l0a, l0b, l0acnt, l0bcnt, splitn, is_init)
        return
    if splitk is not None:
        matmul_splitk(dst, l1a, l1b, l0a, l0b, l0acnt, l0bcnt, splitk, is_init)
        return
    matmul_nosplit(dst, l1a, l1b, l0a, l0b, l0acnt, l0bcnt, is_init)


__all__ = ["matmul"]
