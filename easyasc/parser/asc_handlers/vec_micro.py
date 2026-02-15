from .common import Tensor, Var, dtype_to_cpp, value_to_cpp


def handle_call_micro(inst, helper, expr_map) -> None:
    name = inst.kwargs.get("name", None)
    if not isinstance(name, str):
        raise TypeError(f"call_micro需要name为str类型，当前类型: {type(name)}")

    args = inst.kwargs.get("args", None)
    if args is None:
        args = []
    if not isinstance(args, (list, tuple)):
        raise TypeError(f"call_micro需要args为list或tuple，当前类型: {type(args)}")

    arg_exprs = []
    for arg in args:
        if isinstance(arg, Tensor):
            dtype = dtype_to_cpp(arg.dtype)
            expr = value_to_cpp(arg, expr_map)
            arg_exprs.append(f"(__ubuf__ {dtype}*) ({expr}).GetPhyAddr()")
        elif isinstance(arg, Var):
            arg_exprs.append(value_to_cpp(arg, expr_map))
        else:
            raise TypeError(f"call_micro仅支持Tensor或Var参数，当前类型: {type(arg)}")

    helper(f"{name}({', '.join(arg_exprs)});")
