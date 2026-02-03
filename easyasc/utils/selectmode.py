class SelectModeType:
    """选择模式值类，用于表示具体的选择模式"""
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
            raise TypeError(f"无法比较SelectModeType与{type(other)}")
        return self.name == other.name


class SelectMode:
    """选择模式枚举类"""
    TENSOR_TENSOR = SelectModeType("TENSOR_TENSOR")
    TENSOR_SCALAR = SelectModeType("TENSOR_SCALAR")
