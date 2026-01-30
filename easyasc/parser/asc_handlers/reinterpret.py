from .common import Tensor, dtype_to_cpp, value_to_cpp


def handle_reinterpret(inst, helper, expr_map) -> None:
    dst = inst.kwargs.get("dst", None)
    if not isinstance(dst, Tensor):
        raise TypeError(f"reinterpret需要Tensor类型，当前类型: {type(dst)}")
    src = inst.kwargs.get("src", None)
    if not isinstance(src, Tensor):
        raise TypeError(f"reinterpret需要Tensor类型，当前类型: {type(src)}")
    dst_dtype = dtype_to_cpp(dst.dtype)
    src_expr = value_to_cpp(src, expr_map)
    helper(f"LocalTensor<{dst_dtype}> {dst.name} = {src_expr}.ReinterpretCast<{dst_dtype}>();")
