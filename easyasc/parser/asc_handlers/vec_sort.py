from .common import Tensor, value_to_cpp


def handle_vec_sort32(inst, helper, expr_map) -> None:
    dst = inst.kwargs.get("dst", None)
    if not isinstance(dst, Tensor):
        raise TypeError(f"Sort32需要Tensor类型，当前类型: {type(dst)}")
    src = inst.kwargs.get("src", None)
    if not isinstance(src, Tensor):
        raise TypeError(f"Sort32需要Tensor类型，当前类型: {type(src)}")
    idx = inst.kwargs.get("idx", None)
    if not isinstance(idx, Tensor):
        raise TypeError(f"Sort32需要Tensor类型，当前类型: {type(idx)}")
    dst_expr = value_to_cpp(dst, expr_map)
    src_expr = value_to_cpp(src, expr_map)
    idx_expr = value_to_cpp(idx, expr_map)
    repeat = value_to_cpp(inst.kwargs.get("repeat", None), expr_map)
    helper(f"Sort32({dst_expr}, {src_expr}, {idx_expr}, {repeat});")


def handle_vec_mergesort4(inst, helper, expr_map) -> None:
    dst = inst.kwargs.get("dst", None)
    if not isinstance(dst, Tensor):
        raise TypeError(f"Mergesort4需要Tensor类型，当前类型: {type(dst)}")
    src = inst.kwargs.get("src", None)
    if not isinstance(src, Tensor):
        raise TypeError(f"Mergesort4需要Tensor类型，当前类型: {type(src)}")
    dst_expr = value_to_cpp(dst, expr_map)
    src_expr = value_to_cpp(src, expr_map)
    length_per_blk = value_to_cpp(inst.kwargs.get("length_per_blk", None), expr_map)
    repeat = value_to_cpp(inst.kwargs.get("repeat", None), expr_map)
    helper(f"MERGESORT4({dst_expr}, {src_expr}, {length_per_blk}, {repeat});")


def handle_vec_mergesort_2seq(inst, helper, expr_map) -> None:
    dst = inst.kwargs.get("dst", None)
    if not isinstance(dst, Tensor):
        raise TypeError(f"Mergesort2Seq需要Tensor类型，当前类型: {type(dst)}")
    src1 = inst.kwargs.get("src1", None)
    if not isinstance(src1, Tensor):
        raise TypeError(f"Mergesort2Seq需要Tensor类型，当前类型: {type(src1)}")
    src2 = inst.kwargs.get("src2", None)
    if not isinstance(src2, Tensor):
        raise TypeError(f"Mergesort2Seq需要Tensor类型，当前类型: {type(src2)}")
    dst_expr = value_to_cpp(dst, expr_map)
    src1_expr = value_to_cpp(src1, expr_map)
    src2_expr = value_to_cpp(src2, expr_map)
    size1 = value_to_cpp(inst.kwargs.get("size1", None), expr_map)
    size2 = value_to_cpp(inst.kwargs.get("size2", None), expr_map)
    helper(f"MERGESORT2SEQ({dst_expr}, {src1_expr}, {src2_expr}, {size1}, {size2});")
