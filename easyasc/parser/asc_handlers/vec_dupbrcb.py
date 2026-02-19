from .common import Tensor, value_to_cpp, dtype_to_cpp


def handle_vec_dup(inst, helper, expr_map) -> None:
    dst = inst.kwargs.get("dst", None)
    if not isinstance(dst, Tensor):
        raise TypeError(f"Duplicate requires Tensor type, current type: {type(dst)}")
    src = inst.kwargs.get("src", None)
    dst_expr = value_to_cpp(dst, expr_map)
    src_expr = value_to_cpp(src, expr_map)
    repeat = value_to_cpp(inst.kwargs.get("repeat", None), expr_map)
    dst_blk_stride = value_to_cpp(inst.kwargs.get("dst_blk_stride", None), expr_map)
    dst_rep_stride = value_to_cpp(inst.kwargs.get("dst_rep_stride", None), expr_map)
    dtype = dtype_to_cpp(dst.dtype)
    helper(
        f"Duplicate<{dtype}, false>({dst_expr}, ({dtype}){src_expr}, MASK_PLACEHOLDER, {repeat}, {dst_blk_stride}, {dst_rep_stride});"
    )


def handle_vec_brcb(inst, helper, expr_map) -> None:
    dst = inst.kwargs.get("dst", None)
    if not isinstance(dst, Tensor):
        raise TypeError(f"Brcb requires Tensor type, current type: {type(dst)}")
    src = inst.kwargs.get("src", None)
    if not isinstance(src, Tensor):
        raise TypeError(f"Brcb requires Tensor type, current type: {type(src)}")
    dst_expr = value_to_cpp(dst, expr_map)
    src_expr = value_to_cpp(src, expr_map)
    repeat = value_to_cpp(inst.kwargs.get("repeat", None), expr_map)
    dst_blk_stride = value_to_cpp(inst.kwargs.get("dst_blk_stride", None), expr_map)
    dst_rep_stride = value_to_cpp(inst.kwargs.get("dst_rep_stride", None), expr_map)
    helper(f"Brcb({dst_expr}, {src_expr}, {repeat}, {{{dst_blk_stride}, {dst_rep_stride}}});")
