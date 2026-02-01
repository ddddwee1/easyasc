from .common import Var, value_to_cpp


def handle_start_loop(inst, helper, expr_map) -> None:
    var = inst.kwargs.get("var", None)
    if not isinstance(var, Var):
        raise TypeError(f"start_loop需要Var类型，当前类型: {type(var)}")
    start = value_to_cpp(inst.kwargs.get("start", None), expr_map)
    stop = value_to_cpp(inst.kwargs.get("stop", None), expr_map)
    step = value_to_cpp(inst.kwargs.get("step", None), expr_map)
    helper(f"for ({var.name} = {start}; {var.name} < {stop}; {var.name} += {step}) {{")
    helper.ir()


def handle_end_loop(inst, helper, expr_map) -> None:
    helper.il()
    helper("}")


def handle_start_if(inst, helper, expr_map) -> None:
    cond = value_to_cpp(inst.kwargs.get("cond", None), expr_map)
    helper(f"if ({cond}) {{")
    helper.ir()


def handle_start_elif(inst, helper, expr_map) -> None:
    cond = value_to_cpp(inst.kwargs.get("cond", None), expr_map)
    helper.il()
    helper(f"}} else if ({cond}) {{")
    helper.ir()


def handle_start_else(inst, helper, expr_map) -> None:
    helper.il()
    helper("} else {")
    helper.ir()


def handle_end_if(inst, helper, expr_map) -> None:
    helper.il()
    helper("}")
