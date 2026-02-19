class SelectModeType:
    """Select mode value class representing a concrete select mode."""
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"SelectModeType('{self.name}')"

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, SelectModeType):
            raise TypeError(f"Cannot compare SelectModeType with {type(other)}")
        return self.name == other.name


class SelectMode:
    """Select mode enum-like class."""
    TENSOR_TENSOR = SelectModeType("TENSOR_TENSOR")
    TENSOR_SCALAR = SelectModeType("TENSOR_SCALAR")
