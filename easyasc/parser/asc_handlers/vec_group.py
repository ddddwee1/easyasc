from .common import Tensor, dtype_to_cpp, value_to_cpp


def _handle_vec_group(inst, helper, expr_map) -> dict:
    dst = inst.kwargs.get("dst", None)
    if not isinstance(dst, Tensor):
        raise TypeError(f"Group需要Tensor类型，当前类型: {type(dst)}")
    src = inst.kwargs.get("src", None)
    if not isinstance(src, Tensor):
        raise TypeError(f"Group需要Tensor类型，当前类型: {type(src)}")
    dst_expr = value_to_cpp(dst, expr_map)
    src_expr = value_to_cpp(src, expr_map)
    repeat = value_to_cpp(inst.kwargs.get("repeat", None), expr_map)
    dst_rep_stride = value_to_cpp(inst.kwargs.get("dst_rep_stride", None), expr_map)
    src_blk_stride = value_to_cpp(inst.kwargs.get("src_blk_stride", None), expr_map)
    src_rep_stride = value_to_cpp(inst.kwargs.get("src_rep_stride", None), expr_map)
    dtype = dtype_to_cpp(dst.dtype)
    return {
        "dst_expr": dst_expr,
        "src_expr": src_expr,
        "repeat": repeat,
        "dst_rep_stride": dst_rep_stride,
        "src_blk_stride": src_blk_stride,
        "src_rep_stride": src_rep_stride,
        "dtype": dtype,
    }


def handle_vec_vcadd(inst, helper, expr_map) -> None:
    args = _handle_vec_group(inst, helper, expr_map)
    helper(
        f"WholeReduceSum<{args['dtype']}, false>({args['dst_expr']}, {args['src_expr']}, "
        f"MASK_PLACEHOLDER, {args['repeat']}, {args['dst_rep_stride']}, "
        f"{args['src_blk_stride']}, {args['src_rep_stride']});"
    )


def handle_vec_vcgadd(inst, helper, expr_map) -> None:
    args = _handle_vec_group(inst, helper, expr_map)
    helper(
        f"BlockReduceSum<{args['dtype']}, false>({args['dst_expr']}, {args['src_expr']}, "
        f"{args['repeat']}, MASK_PLACEHOLDER, {args['dst_rep_stride']}, "
        f"{args['src_blk_stride']}, {args['src_rep_stride']});"
    )


def handle_vec_vcpadd(inst, helper, expr_map) -> None:
    args = _handle_vec_group(inst, helper, expr_map)
    helper(
        f"PairReduceSum<{args['dtype']}, false>({args['dst_expr']}, {args['src_expr']}, "
        f"{args['repeat']}, MASK_PLACEHOLDER, {args['dst_rep_stride']}, "
        f"{args['src_blk_stride']}, {args['src_rep_stride']});"
    )


def handle_vec_vcmax(inst, helper, expr_map) -> None:
    args = _handle_vec_group(inst, helper, expr_map)
    helper(
        f"WholeReduceMax<{args['dtype']}, false>({args['dst_expr']}, {args['src_expr']}, "
        f"MASK_PLACEHOLDER, {args['repeat']}, {args['dst_rep_stride']}, "
        f"{args['src_blk_stride']}, {args['src_rep_stride']}, ReduceOrder::ORDER_ONLY_VALUE);"
    )


def handle_vec_vcgmax(inst, helper, expr_map) -> None:
    args = _handle_vec_group(inst, helper, expr_map)
    helper(
        f"BlockReduceMax<{args['dtype']}, false>({args['dst_expr']}, {args['src_expr']}, "
        f"{args['repeat']}, MASK_PLACEHOLDER, {args['dst_rep_stride']}, "
        f"{args['src_blk_stride']}, {args['src_rep_stride']});"
    )


def handle_vec_vcmin(inst, helper, expr_map) -> None:
    args = _handle_vec_group(inst, helper, expr_map)
    helper(
        f"WholeReduceMin<{args['dtype']}, false>({args['dst_expr']}, {args['src_expr']}, "
        f"MASK_PLACEHOLDER, {args['repeat']}, {args['dst_rep_stride']}, "
        f"{args['src_blk_stride']}, {args['src_rep_stride']}, ReduceOrder::ORDER_ONLY_VALUE);"
    )


def handle_vec_vcgmin(inst, helper, expr_map) -> None:
    args = _handle_vec_group(inst, helper, expr_map)
    helper(
        f"BlockReduceMin<{args['dtype']}, false>({args['dst_expr']}, {args['src_expr']}, "
        f"{args['repeat']}, MASK_PLACEHOLDER, {args['dst_rep_stride']}, "
        f"{args['src_blk_stride']}, {args['src_rep_stride']});"
    )
