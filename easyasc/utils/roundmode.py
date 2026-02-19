class RoundModeType:
    """Round mode value class representing a concrete round mode."""
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"RoundModeType('{self.name}')"

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, RoundModeType):
            raise TypeError(f"Cannot compare RoundModeType with {type(other)}")
        return self.name == other.name


class RoundMode:
    """Round mode enum-like class."""
    NONE = RoundModeType("CAST_NONE")
    TO_EVEN = RoundModeType("CAST_RINT")
    AWAY_FROM_ZERO = RoundModeType("CAST_ROUND")
    FLOOR = RoundModeType("CAST_FLOOR")
    CEIL = RoundModeType("CAST_CEIL")
    TRUNC = RoundModeType("CAST_TRUNC")
