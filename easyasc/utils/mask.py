class MaskTypeValue:
    """Mask mode value class representing a concrete mask mode."""
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"MaskTypeValue('{self.name}')"

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, MaskTypeValue):
            raise TypeError(f"Cannot compare MaskTypeValue with {type(other)}")
        return self.name == other.name


class MaskType:
    """Mask mode enum-like class."""
    ALL = MaskTypeValue("ALL")
    LOWEST1 = MaskTypeValue("VL1")
    LOWEST2 = MaskTypeValue("VL2")
    LOWEST3 = MaskTypeValue("VL3")
    LOWEST4 = MaskTypeValue("VL4")
    LOWEST8 = MaskTypeValue("VL8")
    LOWEST16 = MaskTypeValue("VL16")
    LOWEST32 = MaskTypeValue("VL32")
    LOWEST128 = MaskTypeValue("VL128")
    MULTI3 = MaskTypeValue("M3")
    MULTI4 = MaskTypeValue("M4")
    LOWHALF = MaskTypeValue("H")
    LOWQUAT = MaskTypeValue("Q")
    NONE = MaskTypeValue("ALLF")
