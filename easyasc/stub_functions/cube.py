from typing import Union

from ..utils.Tensor import Tensor, GMTensor
from ..utils.var import Var
from ..utils.instruction import Instruction
from ..utils.positions import Position
from .. import globvars


def _validate_var_or_int(value: object, label: str) -> None:
    if not isinstance(value, (Var, int)):
        raise TypeError(f"{label}必须是Var或int类型，当前类型: {type(value)}")


def gm_to_l1_nd2nz(
    dst: Tensor,
    src: GMTensor,
    M: Union[int, Var, None]=None,
    N: Union[int, Var, None]=None,
    N_src: Union[int, Var, None]=None,
    M_dst: Union[int, Var, None]=None,
):
    """
    Stub function for GM to L1 copy for 2D tensors with non-zero strides.
    """
    if not isinstance(dst, Tensor):
        raise TypeError(f"dst必须是Tensor类型，当前类型: {type(dst)}")
    if dst.position is not Position.L1:
        raise ValueError(f"dst必须在L1位置，当前位置: {dst.position}")
    if not isinstance(src, GMTensor):
        raise TypeError(f"src必须是GMTensor类型，当前类型: {type(src)}")

    if M is None or N is None or N_src is None:
        if not (M is None and N is None and N_src is None):
            raise ValueError("M、N、N_src必须同时为None")
        slice_mask = getattr(src, "slice_mask", None)
        slice_dims = [idx for idx, flag in enumerate(slice_mask or []) if flag]
        if len(slice_dims) == 2:
            first, second = slice_dims[0], slice_dims[1]
            M = src.span[first]
            N = src.span[second]
            N_src = src.shape[second]
        elif len(slice_dims) == 1:
            dim = slice_dims[0]
            M = 1
            N = src.span[dim]
            N_src = src.shape[dim]
        else:
            raise ValueError("src必须包含1或2个slice维度以推断M/N/N_src")

    if M_dst is None:
        M_dst = dst.shape[0]

    _validate_var_or_int(M, "M")
    _validate_var_or_int(N, "N")
    _validate_var_or_int(N_src, "N_src")
    _validate_var_or_int(M_dst, "M_dst")

    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction(
                "gm_to_l1_nd2nz",
                dst=dst,
                src=src,
                M=M,
                N=N,
                N_src=N_src,
                M_dst=M_dst,
            )
        )


def l1_to_l0(dst: Tensor, src: Tensor, m_dst: Union[int, Var, None]=None, n_dst: Union[int, Var, None]=None, m_src: Union[int, Var, None]=None, n_src: Union[int, Var, None]=None):
    if not isinstance(dst, Tensor):
        raise TypeError(f"dst必须是Tensor类型，当前类型: {type(dst)}")
    if dst.position not in (Position.L0A, Position.L0B):
        raise ValueError(f"dst必须在L0A或L0B位置，当前位置: {dst.position}")
    if not isinstance(src, Tensor):
        raise TypeError(f"src必须是Tensor类型，当前类型: {type(src)}")
    if src.position is not Position.L1:
        raise ValueError(f"src必须在L1位置，当前位置: {src.position}")

    if m_dst is None:
        m_dst = src.span[0] if hasattr(src, "span") else src.shape[0]
    if n_dst is None:
        n_dst = src.span[1] if hasattr(src, "span") else src.shape[1]
    if m_src is None:
        m_src = src.shape[0]
    if n_src is None:
        n_src = src.shape[1]

    _validate_var_or_int(m_dst, "m_dst")
    _validate_var_or_int(n_dst, "n_dst")
    _validate_var_or_int(m_src, "m_src")
    _validate_var_or_int(n_src, "n_src")

    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction(
                "l1_to_l0",
                dst=dst,
                src=src,
                m_dst=m_dst,
                n_dst=n_dst,
                m_src=m_src,
                n_src=n_src,
            )
        )


def mmad(dst: Tensor, src_a: Tensor, src_b: Tensor, M: Union[int, Var, None]=None, N: Union[int, Var, None]=None, K: Union[int, Var, None]=None, is_init: bool = True):
    if not isinstance(dst, Tensor):
        raise TypeError(f"dst必须是Tensor类型，当前类型: {type(dst)}")
    if dst.position is not Position.L0C:
        raise ValueError(f"dst必须在L0C位置，当前位置: {dst.position}")
    if not isinstance(src_a, Tensor):
        raise TypeError(f"src_a必须是Tensor类型，当前类型: {type(src_a)}")
    if src_a.position is not Position.L0A:
        raise ValueError(f"src_a必须在L0A位置，当前位置: {src_a.position}")
    if not isinstance(src_b, Tensor):
        raise TypeError(f"src_b必须是Tensor类型，当前类型: {type(src_b)}")
    if src_b.position is not Position.L0B:
        raise ValueError(f"src_b必须在L0B位置，当前位置: {src_b.position}")

    if M is None:
        M = src_a.shape[0]
    if N is None:
        N = src_b.shape[0]
    if K is None:
        K = src_a.shape[1]

    _validate_var_or_int(M, "M")
    _validate_var_or_int(N, "N")
    _validate_var_or_int(K, "K")

    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction(
                "mmad",
                dst=dst,
                src_a=src_a,
                src_b=src_b,
                M=M,
                N=N,
                K=K,
                is_init=is_init,
            )
        )


def l0c_to_gm_nz2nd(dst: GMTensor, src: Tensor, M: Union[int, Var, None]=None, N: Union[int, Var, None]=None, N_dst: Union[int, Var, None]=None, M_src: Union[int, Var, None]=None):
    if not isinstance(dst, GMTensor):
        raise TypeError(f"dst必须是GMTensor类型，当前类型: {type(dst)}")
    if not isinstance(src, Tensor):
        raise TypeError(f"src必须是Tensor类型，当前类型: {type(src)}")
    if src.position is not Position.L0C:
        raise ValueError(f"src必须在L0C位置，当前位置: {src.position}")

    if M is None or N is None or N_dst is None:
        if not (M is None and N is None and N_dst is None):
            raise ValueError("M、N、N_dst必须同时为None")
        slice_mask = getattr(dst, "slice_mask", None)
        slice_dims = [idx for idx, flag in enumerate(slice_mask or []) if flag]
        if len(slice_dims) == 2:
            first, second = slice_dims[0], slice_dims[1]
            M = dst.span[first]
            N = dst.span[second]
            N_dst = dst.shape[second]
        elif len(slice_dims) == 1:
            dim = slice_dims[0]
            M = 1
            N = dst.span[dim]
            N_dst = dst.shape[dim]
        else:
            raise ValueError("src必须包含1或2个slice维度以推断M/N/N_src")
        
    if M_src is None:
        M_src = src.shape[0]

    _validate_var_or_int(M, "M")
    _validate_var_or_int(N, "N")
    _validate_var_or_int(M_src, "M_src")

    if globvars.active_kernel is not None:
        if globvars.atomic_enabled and globvars.atomic_type is not dst.dtype:
            globvars.active_kernel.instructions.append(
                Instruction("set_atomic_type", dtype=dst.dtype)
            )
            globvars.atomic_type = dst.dtype
        globvars.active_kernel.instructions.append(
            Instruction(
                "l0c_to_gm_nz2nd",
                dst=dst,
                src=src,
                M=M,
                N=N,
                N_dst=N_dst,
                M_src=M_src,
            )
        )


def l0c_to_l1(
    dst: Tensor,
    src: Tensor,
    M: Union[int, Var, None] = None,
    N: Union[int, Var, None] = None,
    M_dst: Union[int, Var, None] = None,
    M_src: Union[int, Var, None] = None,
    relu: bool = False,
):
    if not isinstance(dst, Tensor):
        raise TypeError(f"dst必须是Tensor类型，当前类型: {type(dst)}")
    if not isinstance(src, Tensor):
        raise TypeError(f"src必须是Tensor类型，当前类型: {type(src)}")
    if src.position is not Position.L0C:
        raise ValueError(f"src必须在L0C位置，当前位置: {src.position}")
    if dst.position is not Position.L1:
        raise ValueError(f"dst必须在L1位置，当前位置: {dst.position}")

    if M is None:
        M = dst.span[0] if hasattr(dst, "span") else dst.shape[0]
    if N is None:
        N = dst.span[1] if hasattr(dst, "span") else dst.shape[1]
    if M_dst is None:
        M_dst = dst.shape[0]
    if M_src is None:
        M_src = src.shape[0]

    _validate_var_or_int(M, "M")
    _validate_var_or_int(N, "N")
    _validate_var_or_int(M_dst, "M_dst")
    _validate_var_or_int(M_src, "M_src")

    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction(
                "l0c_to_l1",
                dst=dst,
                src=src,
                M=M,
                N=N,
                M_dst=M_dst,
                M_src=M_src,
                relu=relu,
            )
        )
