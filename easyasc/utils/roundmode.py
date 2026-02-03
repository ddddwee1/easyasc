class RoundModeType:
    """舍入模式值类，用于表示具体的舍入模式"""
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
            raise TypeError(f"无法比较RoundModeType与{type(other)}")
        return self.name == other.name


class RoundMode:
    """舍入模式枚举类"""
    NONE = RoundModeType("CAST_NONE")
    TO_EVEN = RoundModeType("CAST_RINT")
    AWAY_FROM_ZERO = RoundModeType("CAST_ROUND")
    FLOOR = RoundModeType("CAST_FLOOR")
    CEIL = RoundModeType("CAST_CEIL")
    TRUNC = RoundModeType("CAST_TRUNC")
