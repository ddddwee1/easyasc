from .common import Tensor, Var, dtype_to_cpp, value_to_cpp

_select_cnt = 0


def handle_vec_select(inst, helper, expr_map) -> None:
    global _select_cnt
    dst = inst.kwargs.get("dst", None)
    if not isinstance(dst, Tensor):
        raise TypeError(f"SELECT需要Tensor类型，当前类型: {type(dst)}")
    selmask = inst.kwargs.get("selmask", None)
    if not isinstance(selmask, (Tensor, Var)):
        raise TypeError(f"SELECT需要Tensor或Var类型，当前类型: {type(selmask)}")
    src1 = inst.kwargs.get("src1", None)
    if not isinstance(src1, Tensor):
        raise TypeError(f"SELECT需要Tensor类型，当前类型: {type(src1)}")
    src2 = inst.kwargs.get("src2", None)

    if isinstance(selmask, Var):
        if selmask.dtype is None:
            raise TypeError("selmask的dtype为None，无法推断")
        selmask_dtype = dtype_to_cpp(selmask.dtype)
    else:
        selmask_dtype = dtype_to_cpp(selmask.dtype)

    dst_expr = value_to_cpp(dst, expr_map)
    selmask_expr = value_to_cpp(selmask, expr_map)
    src1_expr = value_to_cpp(src1, expr_map)
    src2_expr = value_to_cpp(src2, expr_map)
    mode = inst.kwargs.get("mode", None)
    if mode is None:
        mode_name = "TENSOR_TENSOR" if isinstance(src2, Tensor) else "TENSOR_SCALAR"
    else:
        mode_name = value_to_cpp(mode, expr_map)
    repeat = value_to_cpp(inst.kwargs.get("repeat", None), expr_map)
    dst_blk_stride = value_to_cpp(inst.kwargs.get("dst_blk_stride", None), expr_map)
    src1_blk_stride = value_to_cpp(inst.kwargs.get("src1_blk_stride", None), expr_map)
    src2_blk_stride = value_to_cpp(inst.kwargs.get("src2_blk_stride", None), expr_map)
    dst_rep_stride = value_to_cpp(inst.kwargs.get("dst_rep_stride", None), expr_map)
    src1_rep_stride = value_to_cpp(inst.kwargs.get("src1_rep_stride", None), expr_map)
    src2_rep_stride = value_to_cpp(inst.kwargs.get("src2_rep_stride", None), expr_map)
    dst_dtype = dtype_to_cpp(dst.dtype)
    src1_dtype = dtype_to_cpp(src1.dtype)

    helper(f"uint64_t maskfull_{_select_cnt}[2] = {{(uint64_t)-1, (uint64_t)-1}};")
    if isinstance(src2, Tensor):
        helper(
            f"Select<{dst_dtype}, {selmask_dtype}, false>({dst_expr}, {selmask_expr}, {src1_expr}, {src2_expr}, "
            f"SELMODE::VSEL_{mode_name}, maskfull_{_select_cnt}, {repeat}, "
            f"{{(uint8_t){dst_blk_stride}, (uint8_t){src1_blk_stride}, (uint8_t){src2_blk_stride}, "
            f"(uint8_t){dst_rep_stride}, (uint8_t){src1_rep_stride}, (uint8_t){src2_rep_stride}}});"
        )
    else:
        helper(
            f"Select<{dst_dtype}, {selmask_dtype}, false>({dst_expr}, {selmask_expr}, {src1_expr}, "
            f"({src1_dtype}){src2_expr}, SELMODE::VSEL_{mode_name}, maskfull_{_select_cnt}, {repeat}, "
            f"{{(uint8_t){dst_blk_stride}, (uint8_t){src1_blk_stride}, (uint8_t){src2_blk_stride}, "
            f"(uint8_t){dst_rep_stride}, (uint8_t){src1_rep_stride}, (uint8_t){src2_rep_stride}}});"
        )
    _select_cnt += 1
