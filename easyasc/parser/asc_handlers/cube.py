from .common import GMTensor, Tensor, Position, value_to_cpp


def handle_gm_to_l1_nd2nz(inst, helper, expr_map) -> None:
    dst = inst.kwargs.get("dst", None)
    if not isinstance(dst, Tensor):
        raise TypeError(f"gm_to_l1_nd2nz requires Tensor type, current type: {type(dst)}")
    src = inst.kwargs.get("src", None)
    if not isinstance(src, GMTensor):
        raise TypeError(f"gm_to_l1_nd2nz requires GMTensor type, current type: {type(src)}")
    dst_expr = value_to_cpp(dst, expr_map)
    src_expr = value_to_cpp(src, expr_map)
    m = value_to_cpp(inst.kwargs.get("M", None), expr_map)
    n = value_to_cpp(inst.kwargs.get("N", None), expr_map)
    n_src = value_to_cpp(inst.kwargs.get("N_src", None), expr_map)
    m_dst = value_to_cpp(inst.kwargs.get("M_dst", None), expr_map)
    helper(f"GM2L1_ND2NZ({dst_expr}, {src_expr}, {m}, {n}, {n_src}, {m_dst});")


def handle_l1_to_l0(inst, helper, expr_map) -> None:
    dst = inst.kwargs.get("dst", None)
    if not isinstance(dst, Tensor):
        raise TypeError(f"l1_to_l0 requires Tensor type, current type: {type(dst)}")
    src = inst.kwargs.get("src", None)
    if not isinstance(src, Tensor):
        raise TypeError(f"l1_to_l0 requires Tensor type, current type: {type(src)}")
    dst_pos = str(dst.position)
    src_transpose = bool(getattr(src, "is_transpose", False))
    if dst_pos == "L0A":
        opname = "L0NZ2NN" if src_transpose else "L0NZ2ZZ"
    elif dst_pos == "L0B":
        opname = "L0NZ2ZN" if src_transpose else "L0NZ2NZ"
    else:
        raise ValueError(f"l1_to_l0does not support dstposition: {dst.position}")
    dst_expr = value_to_cpp(dst, expr_map)
    src_expr = value_to_cpp(src, expr_map)
    m_dst = value_to_cpp(inst.kwargs.get("m_dst", None), expr_map)
    n_dst = value_to_cpp(inst.kwargs.get("n_dst", None), expr_map)
    m_src = value_to_cpp(inst.kwargs.get("m_src", None), expr_map)
    n_src = value_to_cpp(inst.kwargs.get("n_src", None), expr_map)
    helper(f"{opname}({dst_expr}, {src_expr}, {m_dst}, {n_dst}, {m_src}, {n_src});")


def handle_mmad(inst, helper, expr_map) -> None:
    dst = inst.kwargs.get("dst", None)
    if not isinstance(dst, Tensor):
        raise TypeError(f"mmad requires Tensor type, current type: {type(dst)}")
    src_a = inst.kwargs.get("src_a", None)
    if not isinstance(src_a, Tensor):
        raise TypeError(f"mmad requires Tensor type, current type: {type(src_a)}")
    src_b = inst.kwargs.get("src_b", None)
    if not isinstance(src_b, Tensor):
        raise TypeError(f"mmad requires Tensor type, current type: {type(src_b)}")
    dst_expr = value_to_cpp(dst, expr_map)
    src_a_expr = value_to_cpp(src_a, expr_map)
    src_b_expr = value_to_cpp(src_b, expr_map)
    m = value_to_cpp(inst.kwargs.get("M", None), expr_map)
    k = value_to_cpp(inst.kwargs.get("K", None), expr_map)
    n = value_to_cpp(inst.kwargs.get("N", None), expr_map)
    is_init = value_to_cpp(inst.kwargs.get("is_init", None), expr_map)
    helper(f"MMAD({dst_expr}, {src_a_expr}, {src_b_expr}, {m}, {k}, {n}, {is_init});")


def handle_l0c_to_gm_nz2nd(inst, helper, expr_map) -> None:
    dst = inst.kwargs.get("dst", None)
    if not isinstance(dst, GMTensor):
        raise TypeError(f"l0c_to_gm_nz2nd requires GMTensor type, current type: {type(dst)}")
    src = inst.kwargs.get("src", None)
    if not isinstance(src, Tensor):
        raise TypeError(f"l0c_to_gm_nz2nd requires Tensor type, current type: {type(src)}")
    dst_expr = value_to_cpp(dst, expr_map)
    src_expr = value_to_cpp(src, expr_map)
    m = value_to_cpp(inst.kwargs.get("M", None), expr_map)
    n = value_to_cpp(inst.kwargs.get("N", None), expr_map)
    n_dst = value_to_cpp(inst.kwargs.get("N_dst", None), expr_map)
    m_src = value_to_cpp(inst.kwargs.get("M_src", None), expr_map)
    helper(f"L0C2GM_NZ2ND({dst_expr}, {src_expr}, {m}, {n}, {n_dst}, {m_src});")


def handle_l0c_to_l1(inst, helper, expr_map) -> None:
    dst = inst.kwargs.get("dst", None)
    if not isinstance(dst, Tensor):
        raise TypeError(f"l0c_to_l1 requires Tensor type, current type: {type(dst)}")
    src = inst.kwargs.get("src", None)
    if not isinstance(src, Tensor):
        raise TypeError(f"l0c_to_l1 requires Tensor type, current type: {type(src)}")
    dst_expr = value_to_cpp(dst, expr_map)
    src_expr = value_to_cpp(src, expr_map)
    m = value_to_cpp(inst.kwargs.get("M", None), expr_map)
    n = value_to_cpp(inst.kwargs.get("N", None), expr_map)
    m_dst = value_to_cpp(inst.kwargs.get("M_dst", None), expr_map)
    m_src = value_to_cpp(inst.kwargs.get("M_src", None), expr_map)
    relu = value_to_cpp(inst.kwargs.get("relu", None), expr_map)
    helper(f"L0C2L1({dst_expr}, {src_expr}, {m}, {n}, {m_dst}, {m_src}, {relu});")
