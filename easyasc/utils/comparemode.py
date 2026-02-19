class CompareModeType:
    """Compare mode value class representing a concrete compare mode."""
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"CompareModeType('{self.name}')"

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, CompareModeType):
            raise TypeError(f"Cannot compare CompareModeType with {type(other)}")
        return self.name == other.name


class CompareMode:
    """Compare mode enum-like class."""
    LT = CompareModeType("LT")
    GT = CompareModeType("GT")
    GE = CompareModeType("GE")
    LE = CompareModeType("LE")
    EQ = CompareModeType("EQ")
    NE = CompareModeType("NE")
