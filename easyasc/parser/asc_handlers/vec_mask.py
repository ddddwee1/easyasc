from .common import value_to_cpp


def handle_set_mask(inst, helper, expr_map) -> None:
    low = value_to_cpp(inst.kwargs.get("low", None), expr_map)
    high = value_to_cpp(inst.kwargs.get("high", None), expr_map)
    helper(f"SetVectorMask<half, MaskMode::NORMAL>({high}, {low});")


def handle_reset_mask(inst, helper, expr_map) -> None:
    helper("ResetMask();")
