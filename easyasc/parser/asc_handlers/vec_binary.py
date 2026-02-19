from .common import Tensor, dtype_to_cpp, value_to_cpp


def _handle_vec_binary(inst, helper, expr_map, func_name: str) -> None:
    dst = inst.kwargs.get("dst", None)
    if not isinstance(dst, Tensor):
        raise TypeError(f"{func_name}requires Tensor type, current type: {type(dst)}")
    src1 = inst.kwargs.get("src1", None)
    if not isinstance(src1, Tensor):
        raise TypeError(f"{func_name}requires Tensor type, current type: {type(src1)}")
    src2 = inst.kwargs.get("src2", None)
    if not isinstance(src2, Tensor):
        raise TypeError(f"{func_name}requires Tensor type, current type: {type(src2)}")

    dst_expr = value_to_cpp(dst, expr_map)
    src1_expr = value_to_cpp(src1, expr_map)
    src2_expr = value_to_cpp(src2, expr_map)
    repeat = value_to_cpp(inst.kwargs.get("repeat", None), expr_map)
    dst_blk_stride = value_to_cpp(inst.kwargs.get("dst_blk_stride", None), expr_map)
    src1_blk_stride = value_to_cpp(inst.kwargs.get("src1_blk_stride", None), expr_map)
    src2_blk_stride = value_to_cpp(inst.kwargs.get("src2_blk_stride", None), expr_map)
    dst_rep_stride = value_to_cpp(inst.kwargs.get("dst_rep_stride", None), expr_map)
    src1_rep_stride = value_to_cpp(inst.kwargs.get("src1_rep_stride", None), expr_map)
    src2_rep_stride = value_to_cpp(inst.kwargs.get("src2_rep_stride", None), expr_map)
    dtype = dtype_to_cpp(dst.dtype)
    helper(
        f"{func_name}<{dtype}, false>({dst_expr}, {src1_expr}, {src2_expr}, MASK_PLACEHOLDER, {repeat}, "
        f"{{(uint8_t){dst_blk_stride}, (uint8_t){src1_blk_stride}, (uint8_t){src2_blk_stride}, "
        f"(uint8_t){dst_rep_stride}, (uint8_t){src1_rep_stride}, (uint8_t){src2_rep_stride}}});"
    )


def handle_vec_add(inst, helper, expr_map) -> None:
    _handle_vec_binary(inst, helper, expr_map, "Add")


def handle_vec_sub(inst, helper, expr_map) -> None:
    _handle_vec_binary(inst, helper, expr_map, "Sub")


def handle_vec_mul(inst, helper, expr_map) -> None:
    _handle_vec_binary(inst, helper, expr_map, "Mul")


def handle_vec_div(inst, helper, expr_map) -> None:
    _handle_vec_binary(inst, helper, expr_map, "Div")


def handle_vec_max(inst, helper, expr_map) -> None:
    _handle_vec_binary(inst, helper, expr_map, "Max")


def handle_vec_min(inst, helper, expr_map) -> None:
    _handle_vec_binary(inst, helper, expr_map, "Min")


def handle_vec_and(inst, helper, expr_map) -> None:
    _handle_vec_binary(inst, helper, expr_map, "And")


def handle_vec_or(inst, helper, expr_map) -> None:
    _handle_vec_binary(inst, helper, expr_map, "Or")


def handle_vec_muladddst(inst, helper, expr_map) -> None:
    dst = inst.kwargs.get("dst", None)
    if not isinstance(dst, Tensor):
        raise TypeError(f"MulAddDst requires Tensor type, current type: {type(dst)}")
    src1 = inst.kwargs.get("src1", None)
    if not isinstance(src1, Tensor):
        raise TypeError(f"MulAddDst requires Tensor type, current type: {type(src1)}")
    src2 = inst.kwargs.get("src2", None)
    if not isinstance(src2, Tensor):
        raise TypeError(f"MulAddDst requires Tensor type, current type: {type(src2)}")

    dst_expr = value_to_cpp(dst, expr_map)
    src1_expr = value_to_cpp(src1, expr_map)
    src2_expr = value_to_cpp(src2, expr_map)
    repeat = value_to_cpp(inst.kwargs.get("repeat", None), expr_map)
    dst_blk_stride = value_to_cpp(inst.kwargs.get("dst_blk_stride", None), expr_map)
    src1_blk_stride = value_to_cpp(inst.kwargs.get("src1_blk_stride", None), expr_map)
    src2_blk_stride = value_to_cpp(inst.kwargs.get("src2_blk_stride", None), expr_map)
    dst_rep_stride = value_to_cpp(inst.kwargs.get("dst_rep_stride", None), expr_map)
    src1_rep_stride = value_to_cpp(inst.kwargs.get("src1_rep_stride", None), expr_map)
    src2_rep_stride = value_to_cpp(inst.kwargs.get("src2_rep_stride", None), expr_map)
    dst_dtype = dtype_to_cpp(dst.dtype)
    src1_dtype = dtype_to_cpp(src1.dtype)
    helper(
        f"MulAddDst<{dst_dtype}, {src1_dtype}, false>({dst_expr}, {src1_expr}, {src2_expr}, MASK_PLACEHOLDER, {repeat}, "
        f"{{(uint8_t){dst_blk_stride}, (uint8_t){src1_blk_stride}, (uint8_t){src2_blk_stride}, "
        f"(uint8_t){dst_rep_stride}, (uint8_t){src1_rep_stride}, (uint8_t){src2_rep_stride}}});"
    )
