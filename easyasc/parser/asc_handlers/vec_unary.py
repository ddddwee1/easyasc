from .common import Tensor, dtype_to_cpp, value_to_cpp


def _handle_vec_unary(inst, helper, expr_map, func_name: str) -> None:
    dst = inst.kwargs.get("dst", None)
    if not isinstance(dst, Tensor):
        raise TypeError(f"{func_name}需要Tensor类型，当前类型: {type(dst)}")
    src = inst.kwargs.get("src", None)
    if not isinstance(src, Tensor):
        raise TypeError(f"{func_name}需要Tensor类型，当前类型: {type(src)}")

    dst_expr = value_to_cpp(dst, expr_map)
    src_expr = value_to_cpp(src, expr_map)
    repeat = value_to_cpp(inst.kwargs.get("repeat", None), expr_map)
    dst_blk_stride = value_to_cpp(inst.kwargs.get("dst_blk_stride", None), expr_map)
    src_blk_stride = value_to_cpp(inst.kwargs.get("src_blk_stride", None), expr_map)
    dst_rep_stride = value_to_cpp(inst.kwargs.get("dst_rep_stride", None), expr_map)
    src_rep_stride = value_to_cpp(inst.kwargs.get("src_rep_stride", None), expr_map)
    dtype = dtype_to_cpp(dst.dtype)
    helper(
        f"{func_name}<{dtype}, false>({dst_expr}, {src_expr}, MASK_PLACEHOLDER, {repeat}, "
        f"{{(uint16_t){dst_blk_stride}, (uint16_t){src_blk_stride}, "
        f"(uint8_t){dst_rep_stride}, (uint8_t){src_rep_stride}}});"
    )


def handle_vec_exp(inst, helper, expr_map) -> None:
    _handle_vec_unary(inst, helper, expr_map, "Exp")


def handle_vec_ln(inst, helper, expr_map) -> None:
    _handle_vec_unary(inst, helper, expr_map, "Ln")


def handle_vec_abs(inst, helper, expr_map) -> None:
    _handle_vec_unary(inst, helper, expr_map, "Abs")


def handle_vec_rec(inst, helper, expr_map) -> None:
    _handle_vec_unary(inst, helper, expr_map, "Reciprocal")


def handle_vec_sqrt(inst, helper, expr_map) -> None:
    _handle_vec_unary(inst, helper, expr_map, "Sqrt")


def handle_vec_rsqrt(inst, helper, expr_map) -> None:
    _handle_vec_unary(inst, helper, expr_map, "Rsqrt")


def handle_vec_not(inst, helper, expr_map) -> None:
    _handle_vec_unary(inst, helper, expr_map, "Not")


def handle_vec_relu(inst, helper, expr_map) -> None:
    _handle_vec_unary(inst, helper, expr_map, "Relu")
