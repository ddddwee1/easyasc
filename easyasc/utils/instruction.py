class Instruction:
    """Instruction class that stores operation name and keyword arguments."""
    def __init__(self, opname: str, **kwargs):
        if not isinstance(opname, str):
            raise TypeError(f"opname must be str, got: {type(opname)}")
        self.opname = opname
        self.kwargs = dict(kwargs)

    def __getattr__(self, name: str):
        if name in self.kwargs:
            return self.kwargs[name]
        raise AttributeError(f"{type(self).__name__!s} object has no attribute {name!s}")

    def __repr__(self):
        return f"Instruction(opname={self.opname!r}, kwargs={self.kwargs!r})"
