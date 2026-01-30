import inspect
from typing import List

from ..utils.instruction import Instruction
from ..utils.Tensor import GMTensor
from ..utils.var import Var
from .. import globvars


class KernelBase:
    """Kernel基类，保存名称、函数与执行过的指令列表。"""
    def __init__(self, name: str, func):
        self.name = name
        self.func = func
        self.instructions: List[Instruction] = []

    def __call__(self, *args, **kwargs):
        sig = inspect.signature(self.func)
        bound = sig.bind_partial(*args, **kwargs)
        for param_name, value in bound.arguments.items():
            if isinstance(value, (GMTensor, Var)):
                value.name = param_name
            else:
                raise TypeError(f"kernel入参只能为GMTensor或Var，当前{param_name}类型: {type(value)}")
        globvars.active_kernel = self
        globvars.tmp_idx = 0
        for value in bound.arguments.values():
            if isinstance(value, GMTensor):
                self.instructions.append(
                    Instruction("create_gm_tensor", val=value)
                )
        res = self.func(*args, **kwargs)
        globvars.tmp_idx = 0
        globvars.active_kernel = None
        return res
            

    def print_instructions(self):
        print(f"Kernel {self.name}:")
        if not self.instructions:
            print("  (no instructions)")
            return
        for idx, inst in enumerate(self.instructions):
            print(f"  [{idx}] {inst!r}")

    def dump_asc(self, path: str) -> None:
        from ..parser.asc import translate_split
        cube_code, vec_code = translate_split(self.instructions)
        with open(f"{path}_cube.h", "w") as f:
            f.write(cube_code)
        with open(f"{path}_vec.h", "w") as f:
            f.write(vec_code)
