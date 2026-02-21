def validate_core_idx(core_idx: int) -> int:
    if not isinstance(core_idx, int):
        raise TypeError(f"core_idx must be int, got: {type(core_idx)}")
    if core_idx < 0:
        raise ValueError(f"core_idx must be >= 0, got: {core_idx}")
    return core_idx
