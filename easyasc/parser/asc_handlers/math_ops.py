from .common import Var, format_binop, value_to_cpp


def handle_get_cube_num(inst, helper, expr_map) -> None:
    out = inst.kwargs.get("out", None)
    if not isinstance(out, Var):
        raise TypeError(f"GetCubeNum需要Var类型，当前类型: {type(out)}")
    helper(f"{out.name} = GetBlockNum();")


def handle_get_cube_idx(inst, helper, expr_map) -> None:
    out = inst.kwargs.get("out", None)
    if not isinstance(out, Var):
        raise TypeError(f"GetCubeIdx需要Var类型，当前类型: {type(out)}")
    helper(f"{out.name} = get_block_idx();")


def handle_get_vec_num(inst, helper, expr_map) -> None:
    out = inst.kwargs.get("out", None)
    if not isinstance(out, Var):
        raise TypeError(f"GetVecNum需要Var类型，当前类型: {type(out)}")
    helper(f"{out.name} = GetBlockNum() * 2;")


def handle_get_vec_idx(inst, helper, expr_map) -> None:
    out = inst.kwargs.get("out", None)
    if not isinstance(out, Var):
        raise TypeError(f"GetVecIdx需要Var类型，当前类型: {type(out)}")
    helper(f"{out.name} = GetBlockIdx();")


def handle_get_sub_block_idx(inst, helper, expr_map) -> None:
    out = inst.kwargs.get("out", None)
    if not isinstance(out, Var):
        raise TypeError(f"GetSubBlockIdx需要Var类型，当前类型: {type(out)}")
    helper(f"{out.name} = get_subblockid();")


def handle_ceil_div(inst, helper, expr_map) -> None:
    out = inst.kwargs.get("out", None)
    if not isinstance(out, Var):
        raise TypeError(f"CeilDiv需要Var类型，当前类型: {type(out)}")
    a = value_to_cpp(inst.kwargs.get("a", None), expr_map)
    b = value_to_cpp(inst.kwargs.get("b", None), expr_map)
    helper(f"{out.name} = CeilDiv({a}, {b});")


def handle_min(inst, helper, expr_map) -> None:
    out = inst.kwargs.get("out", None)
    if not isinstance(out, Var):
        raise TypeError(f"Min需要Var类型，当前类型: {type(out)}")
    a = value_to_cpp(inst.kwargs.get("a", None), expr_map)
    b = value_to_cpp(inst.kwargs.get("b", None), expr_map)
    helper(f"{out.name} = Min({a}, {b});")


def handle_max(inst, helper, expr_map) -> None:
    out = inst.kwargs.get("out", None)
    if not isinstance(out, Var):
        raise TypeError(f"Max需要Var类型，当前类型: {type(out)}")
    a = value_to_cpp(inst.kwargs.get("a", None), expr_map)
    b = value_to_cpp(inst.kwargs.get("b", None), expr_map)
    helper(f"{out.name} = Max({a}, {b});")


def _handle_align(inst, helper, expr_map, func_name: str, label: str) -> None:
    out = inst.kwargs.get("out", None)
    if not isinstance(out, Var):
        raise TypeError(f"{label}需要Var类型，当前类型: {type(out)}")
    a = value_to_cpp(inst.kwargs.get("a", None), expr_map)
    helper(f"{out.name} = {func_name}({a});")


def handle_align16(inst, helper, expr_map) -> None:
    _handle_align(inst, helper, expr_map, "Align16B", "Align16")


def handle_align32(inst, helper, expr_map) -> None:
    _handle_align(inst, helper, expr_map, "Align32B", "Align32")


def handle_align64(inst, helper, expr_map) -> None:
    _handle_align(inst, helper, expr_map, "Align64B", "Align64")


def handle_align128(inst, helper, expr_map) -> None:
    _handle_align(inst, helper, expr_map, "Align128B", "Align128")


def handle_align256(inst, helper, expr_map) -> None:
    _handle_align(inst, helper, expr_map, "Align256B", "Align256")


def handle_scalar_sqrt(inst, helper, expr_map) -> None:
    out = inst.kwargs.get("out", None)
    if not isinstance(out, Var):
        raise TypeError(f"scalar_sqrt需要Var类型，当前类型: {type(out)}")
    a = value_to_cpp(inst.kwargs.get("a", None), expr_map)
    helper(f"{out.name} = sqrt({a});")


def handle_mul(inst, helper, expr_map) -> None:
    out = inst.kwargs.get("out", None)
    if not isinstance(out, Var):
        raise TypeError(f"var_mul需要Var类型，当前类型: {type(out)}")
    a = value_to_cpp(inst.kwargs.get("a", None), expr_map)
    b = value_to_cpp(inst.kwargs.get("b", None), expr_map)
    helper(f"{out.name} = {format_binop('*', a, b)};")


def handle_div(inst, helper, expr_map) -> None:
    out = inst.kwargs.get("out", None)
    if not isinstance(out, Var):
        raise TypeError(f"var_div需要Var类型，当前类型: {type(out)}")
    a = value_to_cpp(inst.kwargs.get("a", None), expr_map)
    b = value_to_cpp(inst.kwargs.get("b", None), expr_map)
    helper(f"{out.name} = {format_binop('/', a, b)};")


def handle_add(inst, helper, expr_map) -> None:
    out = inst.kwargs.get("out", None)
    if not isinstance(out, Var):
        raise TypeError(f"var_add需要Var类型，当前类型: {type(out)}")
    a = value_to_cpp(inst.kwargs.get("a", None), expr_map)
    b = value_to_cpp(inst.kwargs.get("b", None), expr_map)
    helper(f"{out.name} = {format_binop('+', a, b)};")


def handle_sub(inst, helper, expr_map) -> None:
    out = inst.kwargs.get("out", None)
    if not isinstance(out, Var):
        raise TypeError(f"var_sub需要Var类型，当前类型: {type(out)}")
    a = value_to_cpp(inst.kwargs.get("a", None), expr_map)
    b = value_to_cpp(inst.kwargs.get("b", None), expr_map)
    helper(f"{out.name} = {format_binop('-', a, b)};")
