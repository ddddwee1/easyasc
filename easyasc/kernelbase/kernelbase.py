import inspect
from typing import List, Union

from ..utils.instruction import Instruction
from ..utils.Tensor import GMTensor
from ..utils.var import Var
from ..utils.mutex import CvMutex, VcMutex
from .. import globvars


class KernelBase:
    """Kernel基类，保存名称、函数与执行过的指令列表。"""
    def __init__(self, name: str, func):
        self.name = name
        self.func = func
        self.instructions: List[Instruction] = []
        self.crosscore_mutex: List[Union[CvMutex, VcMutex]] = []
        self._last_bound_args = {}

    def __call__(self, *args, **kwargs):
        sig = inspect.signature(self.func)
        bound = sig.bind_partial(*args, **kwargs)
        self._last_bound_args = dict(bound.arguments)
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
        head_instructions = []
        tail_instructions = []
        for mutex in self.crosscore_mutex:
            if not isinstance(mutex, (CvMutex, VcMutex)):
                raise TypeError(
                    f"crosscore_mutex元素必须是CvMutex或VcMutex，当前类型: {type(mutex)}"
                )
            if isinstance(mutex, CvMutex):
                for _ in range(mutex.depth):
                    head_instructions.append(
                        Instruction("vec_ready", flag_id=mutex.flag_id, pipe=mutex.dst_end_pipe)
                    )
                    tail_instructions.append(
                        Instruction("wait_vec", flag_id=mutex.flag_id, pipe=mutex.src_start_pipe)
                    )
            else:
                for _ in range(mutex.depth):
                    head_instructions.append(
                        Instruction("cube_ready", flag_id=mutex.flag_id, pipe=mutex.dst_end_pipe)
                    )
                    tail_instructions.append(
                        Instruction("wait_cube", flag_id=mutex.flag_id, pipe=mutex.src_start_pipe)
                    )
        if head_instructions:
            self.instructions[:0] = head_instructions
        if tail_instructions:
            self.instructions.extend(tail_instructions)
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

    def dump_kernel(self, path: str) -> None:
        from ..parser.asc import translate_split
        from ..parser.asc_utils import dtype_to_cpp

        cube_code, vec_code = translate_split(self.instructions)
        sig = inspect.signature(self.func)
        param_names = list(sig.parameters.keys())

        gmtensors = {}
        vars_by_name = {}

        def _collect_var(value):
            if isinstance(value, Var):
                vars_by_name.setdefault(value.name, value)
                return
            if isinstance(value, (list, tuple)):
                for item in value:
                    _collect_var(item)
                return
            if isinstance(value, dict):
                for item in value.values():
                    _collect_var(item)

        for inst in self.instructions:
            if inst.opname == "create_gm_tensor":
                val = inst.kwargs.get("val", None)
                if isinstance(val, GMTensor):
                    gmtensors[val.name] = val
            for value in inst.kwargs.values():
                _collect_var(value)

        gmtensor_params = []
        var_params = []
        for name in param_names:
            bound_val = self._last_bound_args.get(name, None)
            if isinstance(bound_val, GMTensor) or name in gmtensors:
                gmtensor_params.append(f"GM_ADDR {name}_")
            else:
                dtype = None
                if isinstance(bound_val, Var):
                    dtype = bound_val.dtype
                elif name in vars_by_name:
                    dtype = vars_by_name[name].dtype
                var_params.append(f"{dtype_to_cpp(dtype)} {name}")

        params = gmtensor_params + ["GM_ADDR workspace"] + var_params
        param_list = ", ".join(params)

        def _wrap(code: str, suffix: str) -> str:
            header = '#include "tensorutils.h"\n\n'
            fn_name = f"{self.name}_{suffix}"
            body = code.strip("\n")
            if body:
                body = "\n".join(f"    {line}" if line else "" for line in body.splitlines())
                return (
                    f"{header}__aicore__ inline void {fn_name}({param_list}) {{\n"
                    f"    int _offset = 0;\n"
                    f"{body}\n"
                    f"}}\n"
                )
            return (
                f"{header}__aicore__ inline void {fn_name}({param_list}) {{\n"
                f"    int _offset = 0;\n"
                f"}}\n"
            )

        with open(f"{path}_cube.h", "w") as f:
            f.write(_wrap(cube_code, "cube"))
        with open(f"{path}_vec.h", "w") as f:
            f.write(_wrap(vec_code, "vec"))
