from .common import Tensor, value_to_cpp


def handle_vec_gather(inst, helper, expr_map) -> None:
    dst = inst.kwargs.get("dst", None)
    if not isinstance(dst, Tensor):
        raise TypeError(f"Gather需要Tensor类型，当前类型: {type(dst)}")
    src = inst.kwargs.get("src", None)
    if not isinstance(src, Tensor):
        raise TypeError(f"Gather需要Tensor类型，当前类型: {type(src)}")
    offset = inst.kwargs.get("offset", None)
    dst_expr = value_to_cpp(dst, expr_map)
    src_expr = value_to_cpp(src, expr_map)
    offset_expr = value_to_cpp(offset, expr_map)
    idx_expr = value_to_cpp(inst.kwargs.get("idx", None), expr_map)
    repeat = value_to_cpp(inst.kwargs.get("repeat", None), expr_map)
    dst_rep_stride = value_to_cpp(inst.kwargs.get("dst_rep_stride", None), expr_map)
    helper(f"Gather({dst_expr}, {src_expr}, {offset_expr}, {idx_expr}, VECTORFULLMASK, {repeat}, {dst_rep_stride});")


def handle_vec_scatter(inst, helper, expr_map) -> None:
    dst = inst.kwargs.get("dst", None)
    if not isinstance(dst, Tensor):
        raise TypeError(f"Scatter需要Tensor类型，当前类型: {type(dst)}")
    src = inst.kwargs.get("src", None)
    if not isinstance(src, Tensor):
        raise TypeError(f"Scatter需要Tensor类型，当前类型: {type(src)}")
    offset = inst.kwargs.get("offset", None)
    dst_expr = value_to_cpp(dst, expr_map)
    src_expr = value_to_cpp(src, expr_map)
    offset_expr = value_to_cpp(offset, expr_map)
    idx_expr = value_to_cpp(inst.kwargs.get("idx", None), expr_map)
    repeat = value_to_cpp(inst.kwargs.get("repeat", None), expr_map)
    src_rep_stride = value_to_cpp(inst.kwargs.get("src_rep_stride", None), expr_map)
    helper(f"Scatter({dst_expr}, {src_expr}, {offset_expr}, {idx_expr}, VECTORFULLMASK, {repeat}, {src_rep_stride});")
