from typing import Union

from ...utils.Tensor import Tensor, GMTensor
from ...utils.var import Var
from ...utils.positions import Position
from ...utils.instruction import Instruction
from ... import globvars
from .vecutils import validate_var_or_int
from ..var_op import CeilDiv


def gm_to_ub_pad(
    dst: Tensor,
    src: GMTensor,
    n_burst: Union[int, Var, None] = None,
    burst_len_element: Union[int, Var, None] = None,
    src_stride_element: Union[int, Var, None] = None,
    dst_stride: Union[int, Var, None] = None,
) -> None:
    if not isinstance(dst, Tensor):
        raise TypeError(f"dst必须是Tensor类型，当前类型: {type(dst)}")
    if not isinstance(src, GMTensor):
        raise TypeError(f"src必须是GMTensor类型，当前类型: {type(src)}")
    if dst.position is not Position.UB:
        raise ValueError(f"dst必须在UB位置，当前位置: {dst.position}")
    if dst.dtype != src.dtype:
        raise ValueError("dst/src的数据类型必须一致")

    span = src.span if hasattr(src, "span") else src.shape
    span0 = span[0]
    span1 = span[1]
    shape0 = src.shape[0]
    shape1 = dst.shape[1]
    validate_var_or_int(span0, "src.span[0]")
    validate_var_or_int(span1, "src.span[1]")
    validate_var_or_int(shape0, "src.shape[0]")
    validate_var_or_int(shape1, "dst.shape[1]")
    if n_burst is None:
        n_burst = span0
    if burst_len_element is None:
        burst_len_element = span1
    if src_stride_element is None:
        src_stride_element = shape0 - span0
    if dst_stride is None:
        dst_stride = CeilDiv(shape1 - span1, dst.dtype.C0)
    if n_burst is None or burst_len_element is None or src_stride_element is None or dst_stride is None:
        raise ValueError("gm_to_ub_pad参数推断失败")

    validate_var_or_int(n_burst, "n_burst")
    validate_var_or_int(burst_len_element, "burst_len_element")
    validate_var_or_int(src_stride_element, "src_stride_element")
    validate_var_or_int(dst_stride, "dst_stride")

    burst_len_byte = burst_len_element * dst.dtype.size
    src_stride_byte = src_stride_element * dst.dtype.size

    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction(
                "GM2UBPAD",
                dst=dst,
                src=src,
                n_burst=n_burst,
                burst_len_byte=burst_len_byte,
                src_stride_byte=src_stride_byte,
                dst_stride=dst_stride,
            )
        )


def ub_to_gm_pad(
    dst: GMTensor,
    src: Tensor,
    n_burst: Union[int, Var, None] = None,
    burst_len_element: Union[int, Var, None] = None,
    src_stride: Union[int, Var, None] = None,
    dst_stride_element: Union[int, Var, None] = None,
) -> None:
    if not isinstance(dst, GMTensor):
        raise TypeError(f"dst必须是GMTensor类型，当前类型: {type(dst)}")
    if not isinstance(src, Tensor):
        raise TypeError(f"src必须是Tensor类型，当前类型: {type(src)}")
    if src.position is not Position.UB:
        raise ValueError(f"src必须在UB位置，当前位置: {src.position}")
    if dst.dtype != src.dtype:
        raise ValueError("dst/src的数据类型必须一致")

    span = dst.span if hasattr(dst, "span") else dst.shape
    span0 = span[0]
    span1 = span[1]
    shape0 = dst.shape[0]
    shape1 = src.shape[1]
    validate_var_or_int(span0, "dst.span[0]")
    validate_var_or_int(span1, "dst.span[1]")
    validate_var_or_int(shape0, "dst.shape[0]")
    validate_var_or_int(shape1, "src.shape[1]")
    if n_burst is None:
        n_burst = span0
    if burst_len_element is None:
        burst_len_element = span1
    if dst_stride_element is None:
        dst_stride_element = shape0 - span0
    if src_stride is None:
        src_stride = CeilDiv(shape1 - span1, src.dtype.C0)
    if n_burst is None or burst_len_element is None or src_stride is None or dst_stride_element is None:
        raise ValueError("ub_to_gm_pad参数推断失败")

    validate_var_or_int(n_burst, "n_burst")
    validate_var_or_int(burst_len_element, "burst_len_element")
    validate_var_or_int(src_stride, "src_stride")
    validate_var_or_int(dst_stride_element, "dst_stride_element")

    burst_len_byte = burst_len_element * src.dtype.size
    dst_stride_byte = dst_stride_element * src.dtype.size

    if globvars.active_kernel is not None:
        if globvars.atomic_enabled and globvars.atomic_type is not dst.dtype:
            globvars.active_kernel.instructions.append(
                Instruction("SETATOMICTYPE", dtype=dst.dtype)
            )
            globvars.atomic_type = dst.dtype
        globvars.active_kernel.instructions.append(
            Instruction(
                "UB2GMPAD",
                dst=dst,
                src=src,
                n_burst=n_burst,
                burst_len_byte=burst_len_byte,
                src_stride=src_stride,
                dst_stride_byte=dst_stride_byte,
            )
        )


def ub_to_ub(
    dst: Tensor,
    src: Tensor,
    n_burst: Union[int, Var, None] = None,
    burst_len: Union[int, Var, None] = None,
    src_stride: Union[int, Var, None] = None,
    dst_stride: Union[int, Var, None] = None,
) -> None:
    if not isinstance(dst, Tensor):
        raise TypeError(f"dst必须是Tensor类型，当前类型: {type(dst)}")
    if not isinstance(src, Tensor):
        raise TypeError(f"src必须是Tensor类型，当前类型: {type(src)}")
    if dst.position is not Position.UB:
        raise ValueError(f"dst必须在UB位置，当前位置: {dst.position}")
    if src.position is not Position.UB:
        raise ValueError(f"src必须在UB位置，当前位置: {src.position}")
    if dst.dtype != src.dtype:
        raise ValueError("dst/src的数据类型必须一致")

    span = dst.span if hasattr(dst, "span") else dst.shape
    span0 = span[0]
    span1 = span[1]
    dst_shape0 = dst.shape[0]
    src_shape1 = src.shape[1]
    validate_var_or_int(span0, "dst.span[0]")
    validate_var_or_int(span1, "dst.span[1]")
    validate_var_or_int(dst_shape0, "dst.shape[0]")
    validate_var_or_int(src_shape1, "src.shape[1]")
    if n_burst is None:
        n_burst = span0
    if burst_len is None:
        burst_len = CeilDiv(span1, dst.dtype.C0)
    if dst_stride is None:
        dst_stride = CeilDiv(dst_shape0 - span0, dst.dtype.C0)
    if src_stride is None:
        src_stride = CeilDiv(src_shape1 - span1, src.dtype.C0)
    if n_burst is None or burst_len is None or dst_stride is None or src_stride is None:
        raise ValueError("ub_to_ub参数推断失败")

    validate_var_or_int(n_burst, "n_burst")
    validate_var_or_int(burst_len, "burst_len")
    validate_var_or_int(src_stride, "src_stride")
    validate_var_or_int(dst_stride, "dst_stride")

    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction(
                "UB2UB",
                dst=dst,
                src=src,
                n_burst=n_burst,
                burst_len=burst_len,
                src_stride=src_stride,
                dst_stride=dst_stride,
            )
        )
