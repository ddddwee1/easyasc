from .common import Tensor, dtype_to_cpp, value_to_cpp


def handle_vec_compare(inst, helper, expr_map) -> None:
    dst = inst.kwargs.get("dst", None)
    if not isinstance(dst, Tensor):
        raise TypeError(f"compare requires Tensor type, current type: {type(dst)}")
    src1 = inst.kwargs.get("src1", None)
    if not isinstance(src1, Tensor):
        raise TypeError(f"compare requires Tensor type, current type: {type(src1)}")
    src2 = inst.kwargs.get("src2", None)
    if not isinstance(src2, Tensor):
        raise TypeError(f"compare requires Tensor type, current type: {type(src2)}")
    dst_expr = value_to_cpp(dst, expr_map)
    src1_expr = value_to_cpp(src1, expr_map)
    src2_expr = value_to_cpp(src2, expr_map)
    mode = value_to_cpp(inst.kwargs.get("mode", None), expr_map)
    repeat = value_to_cpp(inst.kwargs.get("repeat", None), expr_map)
    dst_blk_stride = value_to_cpp(inst.kwargs.get("dst_blk_stride", None), expr_map)
    src1_blk_stride = value_to_cpp(inst.kwargs.get("src1_blk_stride", None), expr_map)
    src2_blk_stride = value_to_cpp(inst.kwargs.get("src2_blk_stride", None), expr_map)
    dst_rep_stride = value_to_cpp(inst.kwargs.get("dst_rep_stride", None), expr_map)
    src1_rep_stride = value_to_cpp(inst.kwargs.get("src1_rep_stride", None), expr_map)
    src2_rep_stride = value_to_cpp(inst.kwargs.get("src2_rep_stride", None), expr_map)
    dst_dtype = dtype_to_cpp(dst.dtype)
    src_dtype = dtype_to_cpp(src1.dtype)
    helper(
        f"Compare<{src_dtype}, {dst_dtype}, false>({dst_expr}, {src1_expr}, {src2_expr}, CMPMODE::{mode}, MASK_PLACEHOLDER, "
        f"{repeat}, {{(uint8_t){dst_blk_stride}, (uint8_t){src1_blk_stride}, (uint8_t){src2_blk_stride}, "
        f"(uint8_t){dst_rep_stride}, (uint8_t){src1_rep_stride}, (uint8_t){src2_rep_stride}}});"
    )


def handle_vec_compares(inst, helper, expr_map) -> None:
    dst = inst.kwargs.get("dst", None)
    if not isinstance(dst, Tensor):
        raise TypeError(f"compare_scalar requires Tensor type, current type: {type(dst)}")
    src1 = inst.kwargs.get("src1", None)
    if not isinstance(src1, Tensor):
        raise TypeError(f"compare_scalar requires Tensor type, current type: {type(src1)}")
    dst_expr = value_to_cpp(dst, expr_map)
    src1_expr = value_to_cpp(src1, expr_map)
    src2_expr = value_to_cpp(inst.kwargs.get("src2", None), expr_map)
    mode = value_to_cpp(inst.kwargs.get("mode", None), expr_map)
    repeat = value_to_cpp(inst.kwargs.get("repeat", None), expr_map)
    dst_blk_stride = value_to_cpp(inst.kwargs.get("dst_blk_stride", None), expr_map)
    src1_blk_stride = value_to_cpp(inst.kwargs.get("src1_blk_stride", None), expr_map)
    dst_rep_stride = value_to_cpp(inst.kwargs.get("dst_rep_stride", None), expr_map)
    src1_rep_stride = value_to_cpp(inst.kwargs.get("src1_rep_stride", None), expr_map)
    dst_dtype = dtype_to_cpp(dst.dtype)
    src_dtype = dtype_to_cpp(src1.dtype)
    helper(
        f"CompareScalar<{src_dtype}, {dst_dtype}, false>({dst_expr}, {src1_expr}, ({src_dtype}){src2_expr}, CMPMODE::{mode}, "
        f"MASK_PLACEHOLDER, {repeat}, {{{dst_blk_stride}, {src1_blk_stride}, {dst_rep_stride}, {src1_rep_stride}}});"
    )


def handle_vec_set_cmpmask(inst, helper, expr_map) -> None:
    src = inst.kwargs.get("src", None)
    if not isinstance(src, Tensor):
        raise TypeError(f"set_cmpmask requires Tensor type, current type: {type(src)}")
    src_expr = value_to_cpp(src, expr_map)
    helper(f"SetCmpMask({src_expr});")
