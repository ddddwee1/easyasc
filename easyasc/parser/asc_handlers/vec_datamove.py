from .common import GMTensor, Tensor, value_to_cpp


def handle_vec_gm2ubpad(inst, helper, expr_map) -> None:
    dst = inst.kwargs.get("dst", None)
    if not isinstance(dst, Tensor):
        raise TypeError(f"gm_to_ub_pad requires Tensor type, current type: {type(dst)}")
    src = inst.kwargs.get("src", None)
    if not isinstance(src, GMTensor):
        raise TypeError(f"gm_to_ub_pad requires GMTensor type, current type: {type(src)}")
    dst_expr = value_to_cpp(dst, expr_map)
    src_expr = value_to_cpp(src, expr_map)
    n_burst = value_to_cpp(inst.kwargs.get("n_burst", None), expr_map)
    burst_len_byte = value_to_cpp(inst.kwargs.get("burst_len_byte", None), expr_map)
    src_stride_byte = value_to_cpp(inst.kwargs.get("src_stride_byte", None), expr_map)
    dst_stride = value_to_cpp(inst.kwargs.get("dst_stride", None), expr_map)
    helper(f"GM2UBPAD({dst_expr}, {src_expr}, {n_burst}, {burst_len_byte}, {src_stride_byte}, {dst_stride});")


def handle_vec_ub2gmpad(inst, helper, expr_map) -> None:
    dst = inst.kwargs.get("dst", None)
    if not isinstance(dst, GMTensor):
        raise TypeError(f"ub_to_gm_pad requires GMTensor type, current type: {type(dst)}")
    src = inst.kwargs.get("src", None)
    if not isinstance(src, Tensor):
        raise TypeError(f"ub_to_gm_pad requires Tensor type, current type: {type(src)}")
    dst_expr = value_to_cpp(dst, expr_map)
    src_expr = value_to_cpp(src, expr_map)
    n_burst = value_to_cpp(inst.kwargs.get("n_burst", None), expr_map)
    burst_len_byte = value_to_cpp(inst.kwargs.get("burst_len_byte", None), expr_map)
    src_stride = value_to_cpp(inst.kwargs.get("src_stride", None), expr_map)
    dst_stride_byte = value_to_cpp(inst.kwargs.get("dst_stride_byte", None), expr_map)
    helper(f"UB2GMPAD({dst_expr}, {src_expr}, {n_burst}, {burst_len_byte}, {src_stride}, {dst_stride_byte});")


def handle_vec_ub2ub(inst, helper, expr_map) -> None:
    dst = inst.kwargs.get("dst", None)
    if not isinstance(dst, Tensor):
        raise TypeError(f"ub_to_ub requires Tensor type, current type: {type(dst)}")
    src = inst.kwargs.get("src", None)
    if not isinstance(src, Tensor):
        raise TypeError(f"ub_to_ub requires Tensor type, current type: {type(src)}")
    dst_expr = value_to_cpp(dst, expr_map)
    src_expr = value_to_cpp(src, expr_map)
    n_burst = value_to_cpp(inst.kwargs.get("n_burst", None), expr_map)
    burst_len = value_to_cpp(inst.kwargs.get("burst_len", None), expr_map)
    src_stride = value_to_cpp(inst.kwargs.get("src_stride", None), expr_map)
    dst_stride = value_to_cpp(inst.kwargs.get("dst_stride", None), expr_map)
    helper(f"UB2UB({dst_expr}, {src_expr}, {n_burst}, {burst_len}, {src_stride}, {dst_stride});")
