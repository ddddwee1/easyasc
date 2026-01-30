class DataTypeValue:
    """数据类型值类，用于表示具体的数据类型"""
    def __init__(self, name: str):
        self.name = name
    
    def __repr__(self):
        return f"DataTypeValue('{self.name}')"
    
    def __str__(self):
        return self.name
    
    def __eq__(self, other):
        """判断两个数据类型是否一致"""
        if self is other:
            return True
        if not isinstance(other, DataTypeValue):
            raise TypeError(f"无法比较DataTypeValue与{type(other)}")
        return self.name == other.name

    @property
    def C0(self):
        if self.name == "half":
            return 16
        elif self.name == "float":
            return 8
        elif self.name == "int":
            return 8
        elif self.name == "int8_t":
            return 32
        elif self.name == "uint8_t":
            return 32
        elif self.name == "int16_t":
            return 16
        elif self.name == "uint16_t":
            return 16
        elif self.name == "bfloat16":
            return 16
        elif self.name == "uint32_t":
            return 8
        elif self.name == "uint64_t":
            return 4
        else:
            raise ValueError(f"未知数据类型: {self.name}")

    @property
    def size(self):
        if self.name == "half":
            return 2
        elif self.name == "float":
            return 4
        elif self.name == "int":
            return 4
        elif self.name == "int8_t":
            return 1
        elif self.name == "uint8_t":
            return 1
        elif self.name == "int16_t":
            return 2
        elif self.name == "uint16_t":
            return 2
        elif self.name == "bfloat16":
            return 2
        elif self.name == "uint32_t":
            return 4
        elif self.name == "uint64_t":
            return 8
        else:
            raise ValueError(f"未知数据类型: {self.name}")


class Datatype:
    """数据类型类，包含half和float两个成员"""
    half = DataTypeValue("half")
    float = DataTypeValue("float")
    int = DataTypeValue("int")
    int8 = DataTypeValue("int8_t")
    uint8 = DataTypeValue("uint8_t")
    int16 = DataTypeValue("int16_t")
    uint16 = DataTypeValue("uint16_t")
    bfloat16 = DataTypeValue("bfloat16_t")
    uint32 = DataTypeValue("uint32_t")
    uint64 = DataTypeValue("uint64_t")
