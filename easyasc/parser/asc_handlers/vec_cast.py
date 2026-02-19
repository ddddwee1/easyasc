from .common import Tensor, dtype_to_cpp, value_to_cpp


def handle_vec_cast(inst, helper, expr_map) -> None:
    dst = inst.kwargs.get("dst", None)
    if not isinstance(dst, Tensor):
        raise TypeError(f"cast requires Tensor type, current type: {type(dst)}")
    src = inst.kwargs.get("src", None)
    if not isinstance(src, Tensor):
        raise TypeError(f"cast requires Tensor type, current type: {type(src)}")
    dst_expr = value_to_cpp(dst, expr_map)
    src_expr = value_to_cpp(src, expr_map)
    mode = value_to_cpp(inst.kwargs.get("mode", None), expr_map)
    repeat = value_to_cpp(inst.kwargs.get("repeat", None), expr_map)
    dst_blk_stride = value_to_cpp(inst.kwargs.get("dst_blk_stride", None), expr_map)
    src_blk_stride = value_to_cpp(inst.kwargs.get("src_blk_stride", None), expr_map)
    dst_rep_stride = value_to_cpp(inst.kwargs.get("dst_rep_stride", None), expr_map)
    src_rep_stride = value_to_cpp(inst.kwargs.get("src_rep_stride", None), expr_map)
    dst_dtype = dtype_to_cpp(dst.dtype)
    src_dtype = dtype_to_cpp(src.dtype)
    helper(
        f"Cast<{dst_dtype}, {src_dtype}, false>({dst_expr}, {src_expr}, RoundMode::{mode}, MASK_PLACEHOLDER, {repeat}, "
        f"{{(uint16_t){dst_blk_stride}, (uint16_t){src_blk_stride}, (uint16_t){dst_rep_stride}, (uint16_t){src_rep_stride}}});"
    )
