from typing import Any, Callable, List

from .. import globvars
from ..utils.instruction import Instruction
from ..utils.Tensor import Tensor
from ..utils.var import Var
from ..utils.positions import Position


class MicroModule:
    def __init__(self, name: str, func: Callable[..., Any]) -> None:
        self.func = func
        self.name = name
        self.instructions: List[Any] = []
        self.tmp_idx = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs:
            raise TypeError("MicroModule不支持关键字参数")
        for arg in args:
            if not isinstance(arg, (Tensor, Var)):
                raise TypeError(f"MicroModule仅支持Tensor或Var入参，当前类型: {type(arg)}")
            if isinstance(arg, Tensor) and arg.position is not Position.UB:
                raise ValueError(f"MicroModule仅支持UB上的Tensor入参，当前位置: {arg.position}")
        if globvars.active_kernel is not None:
            globvars.active_kernel.used_micros.add(self)
            globvars.active_kernel.instructions.append(
                Instruction("call_micro", name=self.name, args=list(args))
            )
        globvars.active_micro = self
        self.func(*args, **kwargs)
        globvars.active_micro = None

    def gen_code(self, path: str) -> None:
        if not isinstance(path, str):
            raise TypeError(f"path必须是str类型，当前类型: {type(path)}")
        from ..parser.asc import translate
        code = translate(self.instructions)
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
