from .common import Tensor, dtype_to_cpp, value_to_cpp


def _handle_vec_unary_scalar(inst, helper, expr_map, func_name: str) -> None:
    dst = inst.kwargs.get("dst", None)
    if not isinstance(dst, Tensor):
        raise TypeError(f"{func_name}需要Tensor类型，当前类型: {type(dst)}")
    src = inst.kwargs.get("src", None)
    if not isinstance(src, Tensor):
        raise TypeError(f"{func_name}需要Tensor类型，当前类型: {type(src)}")

    dst_expr = value_to_cpp(dst, expr_map)
    src_expr = value_to_cpp(src, expr_map)
    val = value_to_cpp(inst.kwargs.get("val", None), expr_map)
    repeat = value_to_cpp(inst.kwargs.get("repeat", None), expr_map)
    dst_blk_stride = value_to_cpp(inst.kwargs.get("dst_blk_stride", None), expr_map)
    src_blk_stride = value_to_cpp(inst.kwargs.get("src_blk_stride", None), expr_map)
    dst_rep_stride = value_to_cpp(inst.kwargs.get("dst_rep_stride", None), expr_map)
    src_rep_stride = value_to_cpp(inst.kwargs.get("src_rep_stride", None), expr_map)
    dtype = dtype_to_cpp(dst.dtype)
    helper(
        f"{func_name}<{dtype}, false>({dst_expr}, {src_expr}, ({dtype}){val}, MASK_PLACEHOLDER, {repeat}, "
        f"{{(uint16_t){dst_blk_stride}, (uint16_t){src_blk_stride}, "
        f"(uint8_t){dst_rep_stride}, (uint8_t){src_rep_stride}}});"
    )


def handle_vec_adds(inst, helper, expr_map) -> None:
    _handle_vec_unary_scalar(inst, helper, expr_map, "Adds")


def handle_vec_muls(inst, helper, expr_map) -> None:
    _handle_vec_unary_scalar(inst, helper, expr_map, "Muls")


def handle_vec_maxs(inst, helper, expr_map) -> None:
    _handle_vec_unary_scalar(inst, helper, expr_map, "Maxs")


def handle_vec_mins(inst, helper, expr_map) -> None:
    _handle_vec_unary_scalar(inst, helper, expr_map, "Mins")


def handle_vec_lrelu(inst, helper, expr_map) -> None:
    _handle_vec_unary_scalar(inst, helper, expr_map, "LeakyRelu")


def handle_vec_axpy(inst, helper, expr_map) -> None:
    dst = inst.kwargs.get("dst", None)
    if not isinstance(dst, Tensor):
        raise TypeError(f"Axpy需要Tensor类型，当前类型: {type(dst)}")
    src = inst.kwargs.get("src", None)
    if not isinstance(src, Tensor):
        raise TypeError(f"Axpy需要Tensor类型，当前类型: {type(src)}")

    dst_expr = value_to_cpp(dst, expr_map)
    src_expr = value_to_cpp(src, expr_map)
    val = value_to_cpp(inst.kwargs.get("val", None), expr_map)
    repeat = value_to_cpp(inst.kwargs.get("repeat", None), expr_map)
    dst_blk_stride = value_to_cpp(inst.kwargs.get("dst_blk_stride", None), expr_map)
    src_blk_stride = value_to_cpp(inst.kwargs.get("src_blk_stride", None), expr_map)
    dst_rep_stride = value_to_cpp(inst.kwargs.get("dst_rep_stride", None), expr_map)
    src_rep_stride = value_to_cpp(inst.kwargs.get("src_rep_stride", None), expr_map)
    dst_dtype = dtype_to_cpp(dst.dtype)
    src_dtype = dtype_to_cpp(src.dtype)
    helper(
        f"Axpy<{dst_dtype}, {src_dtype}, false>({dst_expr}, {src_expr}, ({dst_dtype}){val}, MASK_PLACEHOLDER, {repeat}, "
        f"{{(uint16_t){dst_blk_stride}, (uint16_t){src_blk_stride}, (uint8_t){dst_rep_stride}, (uint8_t){src_rep_stride}}});"
    )
