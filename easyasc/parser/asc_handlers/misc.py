def handle_reset_cache(inst, helper, expr_map) -> None:
    helper("pipe_ptr->Reset();")
    helper("OccupyMMTE1Events();")


def handle_sim_print(inst, helper, expr_map) -> None:
    _ = inst
    _ = helper
    _ = expr_map
