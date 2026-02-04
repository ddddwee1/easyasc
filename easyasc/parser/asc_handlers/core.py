from .common import (
    DBuff,
    GMTensor,
    Tensor,
    Var,
    Position,
    build_offset_expr,
    build_offset_expr_nz,
    dtype_to_cpp,
    position_to_cpp,
    value_to_cpp,
)
from ...utils.datatype import DataTypeValue


def handle_create_var(inst, helper, expr_map) -> None:
    val = inst.kwargs.get("val", None)
    if not isinstance(val, Var):
        raise TypeError(f"create_var需要Var类型，当前类型: {type(val)}")
    dtype = dtype_to_cpp(val.dtype)
    init_expr = value_to_cpp(val.value, expr_map)
    helper(f"{dtype} {val.name} = {init_expr};")


def handle_create_dbuf(inst, helper, expr_map) -> None:
    val = inst.kwargs.get("val", None)
    if not isinstance(val, DBuff):
        raise TypeError(f"create_dbuf需要DBuff类型，当前类型: {type(val)}")
    dtype = dtype_to_cpp(val.dtype)
    position = position_to_cpp(val.position)
    helper(f"DBuff<{dtype}, {position}> {val.name};")


def handle_create_tensor(inst, helper, expr_map) -> None:
    val = inst.kwargs.get("val", None)
    if not isinstance(val, Tensor):
        raise TypeError(f"create_tensor需要Tensor类型，当前类型: {type(val)}")
    dtype = dtype_to_cpp(val.dtype)
    position = position_to_cpp(val.position)
    helper(f"LocalTensor<{dtype}> {val.name};")


def handle_create_gm_tensor(inst, helper, expr_map) -> None:
    val = inst.kwargs.get("val", None)
    if not isinstance(val, GMTensor):
        raise TypeError(f"create_gm_tensor需要GMTensor类型，当前类型: {type(val)}")
    dtype = dtype_to_cpp(val.dtype)
    helper(f"GlobalTensor<{dtype}> {val.name};")
    helper(f"{val.name}.SetGlobalBuffer((__gm__ {dtype}*) {val.name}_);")


def handle_split_workspace(inst, helper, expr_map) -> None:
    dtype = inst.kwargs.get("dtype", None)
    if not isinstance(dtype, DataTypeValue):
        raise TypeError(f"split_workspace expects DataTypeValue, got: {type(dtype)}")
    numel = inst.kwargs.get("numel", None)
    name = inst.kwargs.get("name", None)
    if not isinstance(name, str):
        raise TypeError(f"split_workspace expects name as str, got: {type(name)}")
    dtype_cpp = dtype_to_cpp(dtype)
    numel_cpp = value_to_cpp(numel, expr_map)
    helper(f"workspace = ShiftAddr<{dtype_cpp}>(workspace, {numel_cpp}, _offset);")
    helper(f"GlobalTensor<{dtype_cpp}> {name};")
    helper(f"{name}.SetGlobalBuffer((__gm__ {dtype_cpp}*) workspace);")


def handle_get_buf(inst, helper, expr_map) -> None:
    out = inst.kwargs.get("out", None)
    if not isinstance(out, Tensor):
        raise TypeError(f"get_buf需要Tensor类型，当前类型: {type(out)}")
    buf = inst.kwargs.get("buf", None)
    if not isinstance(buf, DBuff):
        raise TypeError(f"get_buf需要DBuff类型，当前类型: {type(buf)}")
    dtype = dtype_to_cpp(out.dtype)
    index = value_to_cpp(inst.kwargs.get("index", None), expr_map)
    helper(f"Tensor<{dtype}> {out.name} = {buf.name}.get({index});")


def handle_slice_gm_tensor(inst, helper, expr_map) -> None:
    out = inst.kwargs.get("out", None)
    if not isinstance(out, GMTensor):
        raise TypeError(f"slice_gm_tensor需要GMTensor类型，当前类型: {type(out)}")
    src = inst.kwargs.get("src", None)
    if not isinstance(src, GMTensor):
        raise TypeError(f"slice_gm_tensor需要GMTensor类型，当前类型: {type(src)}")
    shape = getattr(out, "shape", None)
    offset = getattr(out, "offset", None)
    offset_expr = build_offset_expr(shape, offset, expr_map)
    src_expr = value_to_cpp(src, expr_map)
    dtype = dtype_to_cpp(out.dtype)
    helper(f"GlobalTensor<{dtype}> {out.name} = {src_expr}[{offset_expr}];")


def handle_slice_tensor(inst, helper, expr_map) -> None:
    out = inst.kwargs.get("out", None)
    if not isinstance(out, Tensor):
        raise TypeError(f"slice_tensor需要Tensor类型，当前类型: {type(out)}")
    src = inst.kwargs.get("src", None)
    if not isinstance(src, Tensor):
        raise TypeError(f"slice_tensor需要Tensor类型，当前类型: {type(src)}")
    shape = getattr(out, "shape", None)
    offset = getattr(out, "offset", None)
    if out.position is Position.L1:
        offset_expr = build_offset_expr_nz(shape, offset, out.dtype, expr_map)
    else:
        offset_expr = build_offset_expr(shape, offset, expr_map)
    src_expr = value_to_cpp(src, expr_map)
    dtype = dtype_to_cpp(out.dtype)
    helper(f"Tensor<{dtype}> {out.name} = {src_expr}[{offset_expr}];")
