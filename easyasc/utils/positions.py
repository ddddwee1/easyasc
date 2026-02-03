POSITION_CPP_MAPPING = {
    "L1": "TPosition::A1",
    "L0A": "TPosition::A2",
    "L0B": "TPosition::B2",
    "L0C": "TPosition::CO1",
    "UB": "TPosition::VECCALC"
}


class PositionType:
    """位置类型类，用于表示具体的位置类型"""
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
            raise TypeError(f"无法比较PositionType与{type(other)}")
        return self.name == other.name


class Position:
    """位置枚举类，包含L1/L0A/L0B/L0C/UB成员"""
    L1 = PositionType("L1")
    L0A = PositionType("L0A")
    L0B = PositionType("L0B")
    L0C = PositionType("L0C")
    UB = PositionType("UB")
