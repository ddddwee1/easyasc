from typing import Union, Tuple

from ...utils.Tensor import Tensor
from ...utils.var import Var
from ...utils.datatype import Datatype
from ...utils.positions import Position
from ...utils.instruction import Instruction
from ... import globvars
from .vecutils import infer_repeat, resolve_strides, validate_var_or_int


_vc_warning = True
_vcg_warning = True
_vcp_warning = True


def _warn_vc() -> None:
    global _vc_warning
    if _vc_warning:
        print("Usage reminder: vcg opearators has destination stride in unit of 2bytes for fp16 and 4bytes for fp32")
        _vc_warning = False


def _warn_vcg() -> None:
    global _vcg_warning
    if _vcg_warning:
        print("Usage reminder: vcg opearators has destination stride in unit of 16bytes for fp16 and 32bytes for fp32")
        _vcg_warning = False


def _warn_vcp() -> None:
    global _vcp_warning
    if _vcp_warning:
        print("Usage reminder: vcp opearators has destination stride in unit of 128bytes")
        _vcp_warning = False


def _validate_group_tensors(dst: Tensor, src: Tensor) -> None:
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
    if dst.dtype not in (Datatype.float, Datatype.half):
        raise ValueError(f"不支持的数据类型: {dst.dtype}")


def _resolve_group_strides(
    dst: Tensor,
    src: Tensor,
    dst_rep_stride: Union[int, Var],
    src_blk_stride: Union[int, Var, None],
    src_rep_stride: Union[int, Var, None],
) -> Tuple[Union[int, Var], Union[int, Var], Union[int, Var]]:
    if dst_rep_stride is None:
        raise ValueError("dst_rep_stride不支持None，请显式传入")
    if src_blk_stride is None or src_rep_stride is None:
        auto_blk, auto_rep = resolve_strides(src, None, None)
        if src_blk_stride is None:
            src_blk_stride = auto_blk
        if src_rep_stride is None:
            src_rep_stride = auto_rep
    return dst_rep_stride, src_blk_stride, src_rep_stride


def _emit_group_inst(
    opname: str,
    dst: Tensor,
    src: Tensor,
    repeat: Union[int, Var],
    dst_rep_stride: Union[int, Var],
    src_blk_stride: Union[int, Var],
    src_rep_stride: Union[int, Var],
) -> None:
    if globvars.active_kernel is not None:
        globvars.active_kernel.instructions.append(
            Instruction(
                opname,
                dst=dst,
                src=src,
                repeat=repeat,
                dst_rep_stride=dst_rep_stride,
                src_blk_stride=src_blk_stride,
                src_rep_stride=src_rep_stride,
            )
        )


def cmax(
    dst: Tensor,
    src: Tensor,
    repeat: Union[int, Var, None] = None,
    dst_rep_stride: Union[int, Var] = 1,
    src_blk_stride: Union[int, Var, None] = None,
    src_rep_stride: Union[int, Var, None] = None,
) -> None:
    _validate_group_tensors(dst, src)
    if repeat is None:
        repeat = infer_repeat(src)
    dst_rep_stride, src_blk_stride, src_rep_stride = _resolve_group_strides(
        dst, src, dst_rep_stride, src_blk_stride, src_rep_stride
    )
    validate_var_or_int(repeat, "repeat")
    validate_var_or_int(dst_rep_stride, "dst_rep_stride")
    validate_var_or_int(src_blk_stride, "src_blk_stride")
    validate_var_or_int(src_rep_stride, "src_rep_stride")
    _emit_group_inst("cmax", dst, src, repeat, dst_rep_stride, src_blk_stride, src_rep_stride)
    _warn_vc()


def cgmax(
    dst: Tensor,
    src: Tensor,
    repeat: Union[int, Var, None] = None,
    dst_rep_stride: Union[int, Var] = 1,
    src_blk_stride: Union[int, Var, None] = None,
    src_rep_stride: Union[int, Var, None] = None,
) -> None:
    _validate_group_tensors(dst, src)
    if repeat is None:
        repeat = infer_repeat(src)
    dst_rep_stride, src_blk_stride, src_rep_stride = _resolve_group_strides(
        dst, src, dst_rep_stride, src_blk_stride, src_rep_stride
    )
    validate_var_or_int(repeat, "repeat")
    validate_var_or_int(dst_rep_stride, "dst_rep_stride")
    validate_var_or_int(src_blk_stride, "src_blk_stride")
    validate_var_or_int(src_rep_stride, "src_rep_stride")
    _emit_group_inst("cgmax", dst, src, repeat, dst_rep_stride, src_blk_stride, src_rep_stride)
    _warn_vcg()


def cmin(
    dst: Tensor,
    src: Tensor,
    repeat: Union[int, Var, None] = None,
    dst_rep_stride: Union[int, Var] = 1,
    src_blk_stride: Union[int, Var, None] = None,
    src_rep_stride: Union[int, Var, None] = None,
) -> None:
    _validate_group_tensors(dst, src)
    if repeat is None:
        repeat = infer_repeat(src)
    dst_rep_stride, src_blk_stride, src_rep_stride = _resolve_group_strides(
        dst, src, dst_rep_stride, src_blk_stride, src_rep_stride
    )
    validate_var_or_int(repeat, "repeat")
    validate_var_or_int(dst_rep_stride, "dst_rep_stride")
    validate_var_or_int(src_blk_stride, "src_blk_stride")
    validate_var_or_int(src_rep_stride, "src_rep_stride")
    _emit_group_inst("cmin", dst, src, repeat, dst_rep_stride, src_blk_stride, src_rep_stride)
    _warn_vc()


def cgmin(
    dst: Tensor,
    src: Tensor,
    repeat: Union[int, Var, None] = None,
    dst_rep_stride: Union[int, Var] = 1,
    src_blk_stride: Union[int, Var, None] = None,
    src_rep_stride: Union[int, Var, None] = None,
) -> None:
    _validate_group_tensors(dst, src)
    if repeat is None:
        repeat = infer_repeat(src)
    dst_rep_stride, src_blk_stride, src_rep_stride = _resolve_group_strides(
        dst, src, dst_rep_stride, src_blk_stride, src_rep_stride
    )
    validate_var_or_int(repeat, "repeat")
    validate_var_or_int(dst_rep_stride, "dst_rep_stride")
    validate_var_or_int(src_blk_stride, "src_blk_stride")
    validate_var_or_int(src_rep_stride, "src_rep_stride")
    _emit_group_inst("cgmin", dst, src, repeat, dst_rep_stride, src_blk_stride, src_rep_stride)
    _warn_vcg()


def cadd(
    dst: Tensor,
    src: Tensor,
    repeat: Union[int, Var, None] = None,
    dst_rep_stride: Union[int, Var] = 1,
    src_blk_stride: Union[int, Var, None] = None,
    src_rep_stride: Union[int, Var, None] = None,
) -> None:
    _validate_group_tensors(dst, src)
    if repeat is None:
        repeat = infer_repeat(src)
    dst_rep_stride, src_blk_stride, src_rep_stride = _resolve_group_strides(
        dst, src, dst_rep_stride, src_blk_stride, src_rep_stride
    )
    validate_var_or_int(repeat, "repeat")
    validate_var_or_int(dst_rep_stride, "dst_rep_stride")
    validate_var_or_int(src_blk_stride, "src_blk_stride")
    validate_var_or_int(src_rep_stride, "src_rep_stride")
    _emit_group_inst("cadd", dst, src, repeat, dst_rep_stride, src_blk_stride, src_rep_stride)
    _warn_vc()


def cgadd(
    dst: Tensor,
    src: Tensor,
    repeat: Union[int, Var, None] = None,
    dst_rep_stride: Union[int, Var] = 1,
    src_blk_stride: Union[int, Var, None] = None,
    src_rep_stride: Union[int, Var, None] = None,
) -> None:
    _validate_group_tensors(dst, src)
    if repeat is None:
        repeat = infer_repeat(src)
    dst_rep_stride, src_blk_stride, src_rep_stride = _resolve_group_strides(
        dst, src, dst_rep_stride, src_blk_stride, src_rep_stride
    )
    validate_var_or_int(repeat, "repeat")
    validate_var_or_int(dst_rep_stride, "dst_rep_stride")
    validate_var_or_int(src_blk_stride, "src_blk_stride")
    validate_var_or_int(src_rep_stride, "src_rep_stride")
    _emit_group_inst("cgadd", dst, src, repeat, dst_rep_stride, src_blk_stride, src_rep_stride)
    _warn_vcg()


def cpadd(
    dst: Tensor,
    src: Tensor,
    repeat: Union[int, Var, None] = None,
    dst_rep_stride: Union[int, Var] = 1,
    src_blk_stride: Union[int, Var, None] = None,
    src_rep_stride: Union[int, Var, None] = None,
) -> None:
    _validate_group_tensors(dst, src)
    if repeat is None:
        repeat = infer_repeat(src)
    dst_rep_stride, src_blk_stride, src_rep_stride = _resolve_group_strides(
        dst, src, dst_rep_stride, src_blk_stride, src_rep_stride
    )
    validate_var_or_int(repeat, "repeat")
    validate_var_or_int(dst_rep_stride, "dst_rep_stride")
    validate_var_or_int(src_blk_stride, "src_blk_stride")
    validate_var_or_int(src_rep_stride, "src_rep_stride")
    _emit_group_inst("cpadd", dst, src, repeat, dst_rep_stride, src_blk_stride, src_rep_stride)
    _warn_vcp()
