from .common import dtype_to_cpp


def handle_atomic_add(inst, helper, expr_map) -> None:
    helper("SetAtomicAdd<float>();")


def handle_atomic_max(inst, helper, expr_map) -> None:
    helper("SetAtomicMax<float>();")


def handle_atomic_min(inst, helper, expr_map) -> None:
    helper("SetAtomicMin<float>();")


def handle_set_atomic_type(inst, helper, expr_map) -> None:
    dtype = inst.kwargs.get("dtype", None)
    dtype_expr = dtype_to_cpp(dtype)
    helper(f"SetAtomicType<{dtype_expr}>();")


def handle_atomic_end(inst, helper, expr_map) -> None:
    helper("SetAtomicNone();")
