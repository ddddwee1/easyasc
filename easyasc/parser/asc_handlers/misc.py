def handle_reset_cache(inst, helper, expr_map) -> None:
    helper("_pipe->Reset();")
