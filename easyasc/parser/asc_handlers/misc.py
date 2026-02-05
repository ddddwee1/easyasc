def handle_reset_cache(inst, helper, expr_map) -> None:
    helper("pipe_ptr->Reset();")
    helper("OccupyMMTE1Events();")
