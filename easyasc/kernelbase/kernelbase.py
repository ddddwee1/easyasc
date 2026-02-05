import inspect
import json
import os
import tarfile
from typing import List, Union

from ..utils.instruction import Instruction
from ..utils.Tensor import GMTensor, DBuff
from ..utils.var import Var
from ..utils.mutex import CvMutex, VcMutex
from ..utils.datatype import Datatype
from ..utils.positions import Position
from .. import globvars


class KernelBase:
    """Kernel基类，保存名称、函数与执行过的指令列表。"""
    def __init__(self, name: str, func):
        self.name = name
        self.func = func
        self.instructions: List[Instruction] = []
        self.crosscore_mutex: List[Union[CvMutex, VcMutex]] = []
        self.workspace_shapes: List[Union[list, tuple]] = []
        self._last_bound_args = {}

    def __call__(self, *args, **kwargs):
        sig = inspect.signature(self.func)
        bound = sig.bind_partial(*args, **kwargs)
        self._last_bound_args = dict(bound.arguments)
        self.workspace_shapes = []
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
        self._l0a = DBuff(Datatype.half, [128, 128], position=Position.L0A, name='_l0a')
        self._l0b = DBuff(Datatype.half, [128, 128], position=Position.L0B, name='_l0b')
        self._l0acnt = Var(0, name='_l0acnt')
        self._l0bcnt = Var(0, name='_l0bcnt')
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
        cube_code, vec_code = translate_split(self.instructions, self.name)
        with open(f"{path}_cube.h", "w") as f:
            f.write(cube_code)
        with open(f"{path}_vec.h", "w") as f:
            f.write(vec_code)

    def dump_kernel(self, path: str) -> None:
        from ..parser.asc import translate_split
        from ..parser.asc_utils import dtype_to_cpp

        cube_code, vec_code = translate_split(self.instructions, self.name)
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
        gmtensor_args = []
        var_params = []
        var_args = []
        for name in param_names:
            bound_val = self._last_bound_args.get(name, None)
            if isinstance(bound_val, GMTensor) or name in gmtensors:
                gmtensor_params.append(f"GM_ADDR {name}_")
                gmtensor_args.append(name)
            else:
                dtype = None
                if isinstance(bound_val, Var):
                    dtype = bound_val.dtype
                elif name in vars_by_name:
                    dtype = vars_by_name[name].dtype
                var_params.append(f"{dtype_to_cpp(dtype)} {name}")
                var_args.append(name)

        params = gmtensor_params + ["GM_ADDR workspace"] + var_params
        param_list = ", ".join(params)
        arg_list = ", ".join(gmtensor_args + ["workspace"] + var_args)

        def _wrap(code: str, suffix: str) -> str:
            header = '#pragma once\n#include "tensorutils.h"\n\n'
            fn_name = f"{self.name}_{suffix}"
            body = code.strip("\n")
            if body:
                body = "\n".join(f"    {line}" if line else "" for line in body.splitlines())
                return (
                    f"{header}__aicore__ inline void {fn_name}({param_list}) {{\n"
                    f"    TPipe* pipe_ptr = GetTPipePtr();\n"
                    f"    int _offset = 0;\n"
                    f"{body}\n"
                    f"}}\n"
                )
            return (
                f"{header}__aicore__ inline void {fn_name}({param_list}) {{\n"
                f"    TPipe* pipe_ptr = GetTPipePtr();\n"
                f"    int _offset = 0;\n"
                f"}}\n"
            )

        with open(f"{path}_cube.h", "w") as f:
            f.write(_wrap(cube_code, "cube"))
        with open(f"{path}_vec.h", "w") as f:
            f.write(_wrap(vec_code, "vec"))
        cpp_params = [f"GM_ADDR {name}" for name in param_names
                      if isinstance(self._last_bound_args.get(name, None), GMTensor) or name in gmtensors]
        cpp_params.extend(["GM_ADDR workspace", "GM_ADDR tiling"])
        cpp_param_list = ", ".join(cpp_params)
        tiling_vars = []
        for name in param_names:
            bound_val = self._last_bound_args.get(name, None)
            if isinstance(bound_val, GMTensor) or name in gmtensors:
                continue
            dtype = None
            if isinstance(bound_val, Var):
                dtype = bound_val.dtype
            elif name in vars_by_name:
                dtype = vars_by_name[name].dtype
            tiling_vars.append(f"    {dtype_to_cpp(dtype)} {name} = tiling_data.{name};\n")
        cpp_body = (
            '#include "kernel_operator.h"\n'
            '#include "lib/matmul_intf.h"\n'
            '#include "tensorutils.h"\n'
            f'#include "{path}_cube.h"\n'
            f'#include "{path}_vec.h"\n'
            "\n\n"
            f'extern "C" __global__ __aicore__ void {path}({cpp_param_list}) {{\n'
            "    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);\n"
            "    GET_TILING_DATA(tiling_data, tiling);\n"
            "    PipeBarrier<PIPE_ALL>();\n"
        )
        cpp_body += "".join(tiling_vars)
        cpp_body += (
            f"    if ASCEND_IS_AIC{{\n"
            f"        {self.name}_cube({arg_list});\n"
            f"    }}\n"
            f"    if ASCEND_IS_AIV{{\n"
            f"        {self.name}_vec({arg_list});\n"
            f"    }}\n"
            "\n}\n"
        )
        with open(f"{path}.cpp", "w") as f:
            f.write(cpp_body)

    def generate_op_host(self) -> None:
        sig = inspect.signature(self.func)
        if not self._last_bound_args:
            raise RuntimeError("generate_op_host需要先调用kernel以绑定参数")

        def _to_camel(name: str) -> str:
            parts = [p for p in name.split("_") if p]
            if not parts:
                return name
            return "".join(p[:1].upper() + p[1:] for p in parts)

        param_names = list(sig.parameters.keys())
        var_names = []
        for name in param_names:
            bound_val = self._last_bound_args.get(name, None)
            if isinstance(bound_val, Var):
                var_names.append(name)

        op_name = _to_camel(self.name)
        tiling_name = f"{op_name}TilingData"
        lines = [
            '#include "register/tilingdata_base.h"',
            "",
            "namespace optiling {",
            f"BEGIN_TILING_DATA_DEF({tiling_name})",
        ]
        for var_name in var_names:
            lines.append(f"  TILING_DATA_FIELD_DEF(uint32_t, {var_name});")
        lines.append("END_TILING_DATA_DEF;")
        lines.append("")
        lines.append(f"REGISTER_TILING_DATA_CLASS({op_name}, {tiling_name})")
        lines.append("}")
        content = "\n".join(lines) + "\n"

        with open(f"{self.name}_tiling.h", "w") as f:
            f.write(content)

    def generate_op_project(self, path: str, cann_path: str) -> None:
        if not isinstance(path, str):
            raise TypeError(f"path必须是str类型，当前类型: {type(path)}")
        if not isinstance(cann_path, str):
            raise TypeError(f"cann_path必须是str类型，当前类型: {type(cann_path)}")
        resources_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources"))
        tar_path = os.path.join(resources_dir, "CustomOp.tar.gz")
        if not os.path.isfile(tar_path):
            raise FileNotFoundError(f"未找到CustomOp压缩包: {tar_path}")
        preset_template = os.path.join(resources_dir, "CMakePresets.json")
        if not os.path.isfile(preset_template):
            raise FileNotFoundError(f"未找到CMakePresets.json模板: {preset_template}")
        dst = os.path.abspath(path)
        if os.path.exists(dst):
            if not os.path.isdir(dst):
                raise FileExistsError(f"目标路径已存在且不是目录: {dst}")
            if os.listdir(dst):
                return
        else:
            os.makedirs(dst, exist_ok=False)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(dst)
        with open(preset_template, "r", encoding="utf-8") as f:
            preset_data = json.load(f)
        updated = False
        device_type = getattr(globvars, "device_type", "")
        compute_unit = None
        if isinstance(device_type, str):
            device_type = device_type.lower()
            if device_type.startswith("b"):
                compute_unit = "ascend910b"
            elif device_type.startswith("d"):
                compute_unit = "ascend910_93"
        for preset in preset_data.get("configurePresets", []):
            cache_vars = preset.get("cacheVariables", {})
            asc_var = cache_vars.get("ASCEND_CANN_PACKAGE_PATH", None)
            if isinstance(asc_var, dict):
                asc_var["value"] = cann_path
                updated = True
            if compute_unit is not None:
                compute_var = cache_vars.get("ASCEND_COMPUTE_UNIT", None)
                if isinstance(compute_var, dict):
                    compute_var["value"] = compute_unit
        if not updated:
            raise ValueError("未找到ASCEND_CANN_PACKAGE_PATH配置项")
        preset_out = os.path.join(dst, "CMakePresets.json")
        with open(preset_out, "w", encoding="utf-8") as f:
            json.dump(preset_data, f, indent=4)
            f.write("\n")
