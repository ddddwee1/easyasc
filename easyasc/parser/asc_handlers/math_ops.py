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
