class Instruction:
    """指令类，保存指令名称与参数"""
    def __init__(self, opname: str, **kwargs):
        if not isinstance(opname, str):
            raise TypeError(f"opname必须是str类型，当前类型: {type(opname)}")
        self.opname = opname
        self.kwargs = dict(kwargs)

    def __getattr__(self, name: str):
        if name in self.kwargs:
            return self.kwargs[name]
        raise AttributeError(f"{type(self).__name__!s}对象没有属性 {name!s}")

    def __repr__(self):
        return f"Instruction(opname={self.opname!r}, kwargs={self.kwargs!r})"
