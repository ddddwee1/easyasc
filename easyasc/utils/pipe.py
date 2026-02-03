class PipeType:
    """流水线类型类，用于表示具体的流水线类型"""
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"PipeType('{self.name}')"

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, PipeType):
            raise TypeError(f"无法比较PipeType与{type(other)}")
        return self.name == other.name
    
    def __hash__(self) -> int:
        return hash(self.name)


class Pipe:
    """流水线枚举类，包含MTE2/MTE1/M/V/FIX/MTE3/S成员"""
    MTE2 = PipeType("MTE2")
    MTE1 = PipeType("MTE1")
    M = PipeType("M")
    V = PipeType("V")
    FIX = PipeType("FIX")
    MTE3 = PipeType("MTE3")
    S = PipeType("S")
    ALL = PipeType("ALL")
