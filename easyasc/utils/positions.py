POSITION_CPP_MAPPING = {
    "L1": "TPosition::A1",
    "L0A": "TPosition::A2",
    "L0B": "TPosition::B2",
    "L0C": "TPosition::CO1",
    "UB": "TPosition::VECCALC"
}


class PositionType:
    """Position value class representing a concrete memory position."""
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"PositionType('{self.name}')"

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, PositionType):
            raise TypeError(f"Cannot compare PositionType with {type(other)}")
        return self.name == other.name

    @property
    def cpp(self):
        return POSITION_CPP_MAPPING[self.name]


class Position:
    """Position enum-like class with L1/L0A/L0B/L0C/UB members."""
    L1 = PositionType("L1")
    L0A = PositionType("L0A")
    L0B = PositionType("L0B")
    L0C = PositionType("L0C")
    UB = PositionType("UB")
