class CompareModeType:
    """比较模式值类，用于表示具体的比较模式"""
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
            raise TypeError(f"无法比较CompareModeType与{type(other)}")
        return self.name == other.name


class CompareMode:
    """比较模式枚举类"""
    LT = CompareModeType("LT")
    GT = CompareModeType("GT")
    GE = CompareModeType("GE")
    LE = CompareModeType("LE")
    EQ = CompareModeType("EQ")
    NE = CompareModeType("NE")
