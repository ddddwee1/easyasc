from .common import PipeType, _pipe_name, value_to_cpp


def _handle_pipe_flag_op(inst, helper, expr_map, opname: str) -> None:
    flag_id = value_to_cpp(inst.kwargs.get("flag_id", None), expr_map)
    pipe = inst.kwargs.get("pipe", None)
    if not isinstance(pipe, PipeType):
        raise TypeError(f"{opname}需要PipeType类型，当前类型: {type(pipe)}")
    pipe_expr = _pipe_name(pipe)
    helper(f"{opname}({flag_id}, {pipe_expr});")


def handle_cube_ready(inst, helper, expr_map) -> None:
    _handle_pipe_flag_op(inst, helper, expr_map, "CUBE_READY")


def handle_vec_ready(inst, helper, expr_map) -> None:
    _handle_pipe_flag_op(inst, helper, expr_map, "VEC_READY")


def handle_wait_cube(inst, helper, expr_map) -> None:
    _handle_pipe_flag_op(inst, helper, expr_map, "WAIT_CUBE")


def handle_wait_vec(inst, helper, expr_map) -> None:
    _handle_pipe_flag_op(inst, helper, expr_map, "WAIT_VEC")


def handle_allcube_ready(inst, helper, expr_map) -> None:
    _handle_pipe_flag_op(inst, helper, expr_map, "ALLCUBE_READY")


def handle_allvec_ready(inst, helper, expr_map) -> None:
    _handle_pipe_flag_op(inst, helper, expr_map, "ALLVEC_READY")


def handle_allcube_wait(inst, helper, expr_map) -> None:
    _handle_pipe_flag_op(inst, helper, expr_map, "ALLCUBE_WAIT")


def handle_allvec_wait(inst, helper, expr_map) -> None:
    _handle_pipe_flag_op(inst, helper, expr_map, "ALLVEC_WAIT")


def handle_bar(inst, helper, expr_map) -> None:
    pipe = inst.kwargs.get("pipe", None)
    if not isinstance(pipe, PipeType):
        raise TypeError(f"barrier需要PipeType类型，当前类型: {type(pipe)}")
    pipe_expr = _pipe_name(pipe)
    helper(f"PipeBarrier<{pipe_expr}>();")
