import inspect
import json
import os
import shutil
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
        self._last_output_gmtensors = set()

    def __call__(self, *args, **kwargs):
        def _collect_gmtensors(value, out):
            if isinstance(value, GMTensor):
                out.add(value)
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    _collect_gmtensors(item, out)
                return
            if isinstance(value, dict):
                for item in value.values():
                    _collect_gmtensors(item, out)
                return

        sig = inspect.signature(self.func)
        bound = sig.bind_partial(*args, **kwargs)
        self._last_bound_args = dict(bound.arguments)
        self._last_output_gmtensors = set()
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
        self.instructions.append(
            Instruction("reset_cache")
        )
        res = self.func(*args, **kwargs)
        outputs = set()
        _collect_gmtensors(res, outputs)
        self._last_output_gmtensors = outputs
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

        def _dtype_to_ge(dtype) -> str:
            name = getattr(dtype, "name", None)
            mapping = {
                "half": "ge::DT_FLOAT16",
                "float": "ge::DT_FLOAT",
                "bfloat16_t": "ge::DT_BF16",
                "int": "ge::DT_INT32",
                "int32_t": "ge::DT_INT32",
                "int64_t": "ge::DT_INT64",
                "uint32_t": "ge::DT_UINT32",
                "uint64_t": "ge::DT_UINT64",
                "int8_t": "ge::DT_INT8",
                "uint8_t": "ge::DT_UINT8",
                "int16_t": "ge::DT_INT16",
                "uint16_t": "ge::DT_UINT16",
            }
            return mapping.get(name, "ge::DT_FLOAT16")

        def _attr_method(dtype) -> str:
            name = getattr(dtype, "name", None)
            if name == "float":
                return "Float(0)"
            return "Int(0)"

        param_names = list(sig.parameters.keys())
        var_names = []
        for name in param_names:
            bound_val = self._last_bound_args.get(name, None)
            if isinstance(bound_val, Var):
                var_names.append(name)

        def _workspace_size_expr() -> str:
            if not self.workspace_shapes:
                return "0"

            def _dim_expr(dim) -> str:
                if isinstance(dim, int):
                    return str(dim)
                if isinstance(dim, Var):
                    if dim.name in var_names:
                        return dim.name
                    if dim.value is not None:
                        return str(dim.value)
                    raise ValueError(f"workspace_shape包含未绑定参数的Var: {dim.name!r}")
                raise TypeError(f"workspace_shape元素必须是int或Var，当前类型: {type(dim)}")

            def _shape_expr(shape) -> str:
                if not isinstance(shape, (list, tuple)):
                    raise TypeError(f"workspace_shape必须是list或tuple，当前类型: {type(shape)}")
                factors = []
                for dim in shape:
                    if isinstance(dim, int) and dim == 1:
                        continue
                    factors.append(_dim_expr(dim))
                if not factors:
                    return "1"
                if len(factors) == 1:
                    return factors[0]
                return "(" + " * ".join(factors) + ")"

            terms = [_shape_expr(shape) for shape in self.workspace_shapes]
            if len(terms) == 1:
                return terms[0]
            return " + ".join(terms)

        workspace_size_expr = _workspace_size_expr()

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

        output_gmtensors = set()
        if isinstance(getattr(self, "_last_output_gmtensors", None), set):
            output_gmtensors = self._last_output_gmtensors

        cpp_lines = [
            f'#include "{self.name}_tiling.h"',
            '#include "register/op_def_registry.h"',
            '#include "tiling/tiling_api.h"',
            "",
            "#define SET_ATTR_TO_TILING(name, dtype, idx) \\",
            "    const auto* name##_ptr = attrs->GetAttrPointer<dtype>(idx); \\",
            "    dtype name = *name##_ptr; \\",
            "    tiling.set_##name(name);",
            "#define GET_ATTR_BY_IDX(name, dtype, idx) \\",
            "    const auto* name##_ptr = attrs->GetAttrPointer<dtype>(idx); \\",
            "    dtype name = *name##_ptr;",
            "",
            "namespace optiling {",
            "static ge::graphStatus TilingFunc(gert::TilingContext* context)",
            "{",
            "    // set block size",
            "    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());",
            "    auto core_num = ascendcPlatform.GetCoreNum();   // NOTE: this is vec number!",
            "    context->SetBlockDim(core_num/2);",
            "",
            f"    {tiling_name} tiling;",
            "    auto attrs = context->GetAttrs();",
        ]
        for idx, name in enumerate(var_names):
            cpp_lines.append(f"    SET_ATTR_TO_TILING({name}, uint32_t, {idx});")
        cpp_lines.extend(
            [
                "    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());",
                "    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());",
                "",
                f"    size_t userWorkspaceSize = {workspace_size_expr};",
                "    size_t sysWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());",
                "    size_t *currentWorkspace = context->GetWorkspaceSizes(1);",
                "    currentWorkspace[0] = userWorkspaceSize + sysWorkspaceSize;",
                "    return ge::GRAPH_SUCCESS;",
                "}",
                "}",
                "",
                "namespace ge {",
                "static ge::graphStatus InferShape(gert::InferShapeContext* context)",
                "{",
                "    (void)context;",
                "    return GRAPH_SUCCESS;",
                "}",
                "",
                "static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)",
                "{",
                "    (void)context;",
                "    return GRAPH_SUCCESS;",
                "}",
                "}",
                "",
                "namespace ops {",
                f"class {op_name} : public OpDef {{",
                "public:",
                f"    explicit {op_name}(const char* name) : OpDef(name)",
                "    {",
            ]
        )

        for name in param_names:
            bound_val = self._last_bound_args.get(name, None)
            if not isinstance(bound_val, GMTensor):
                continue
            ge_dt = _dtype_to_ge(bound_val.dtype)
            if bound_val in output_gmtensors:
                cpp_lines.extend(
                    [
                        f'        this->Output("{name}")',
                        "            .ParamType(REQUIRED)",
                        f"            .DataType({{{ge_dt}}})",
                        "            .Format({ge::FORMAT_ND})",
                        "            .UnknownShapeFormat({ge::FORMAT_ND})",
                        "            .InitValue(0);",
                    ]
                )
            else:
                cpp_lines.extend(
                    [
                        f'        this->Input("{name}")',
                        "            .ParamType(REQUIRED)",
                        f"            .DataType({{{ge_dt}}})",
                        "            .Format({ge::FORMAT_ND})",
                        "            .UnknownShapeFormat({ge::FORMAT_ND});",
                    ]
                )

        for name in var_names:
            bound_val = self._last_bound_args.get(name, None)
            dtype = bound_val.dtype if isinstance(bound_val, Var) else None
            attr_method = _attr_method(dtype)
            cpp_lines.extend(
                [
                    f'        this->Attr("{name}")',
                    "            .AttrType(REQUIRED)",
                    f"            .{attr_method};",
                ]
            )

        cpp_lines.extend(
            [
                "        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);",
                "",
                "        this->AICore().SetTiling(optiling::TilingFunc);",
                '        this->AICore().AddConfig("ascend910b");',
                '        this->AICore().AddConfig("ascend910_95");',
            ]
        )
        cpp_lines.extend(
            [
                "    }",
                "};",
                "",
                f"OP_ADD({op_name});",
                "}",
                "",
            ]
        )
        with open(f"{self.name}.cpp", "w") as f:
            f.write("\n".join(cpp_lines))

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

    def generate(self, out_dir: str, cann_path: str) -> None:
        if not isinstance(out_dir, str):
            raise TypeError(f"out_dir必须是str类型，当前类型: {type(out_dir)}")
        if not isinstance(cann_path, str):
            raise TypeError(f"cann_path必须是str类型，当前类型: {type(cann_path)}")

        self.generate_op_project(out_dir, cann_path)

        abs_out_dir = os.path.abspath(out_dir)
        op_host_dir = os.path.join(abs_out_dir, "op_host")
        op_kernel_dir = os.path.join(abs_out_dir, "op_kernel")
        os.makedirs(op_host_dir, exist_ok=True)
        os.makedirs(op_kernel_dir, exist_ok=True)

        cwd = os.getcwd()
        try:
            os.chdir(op_host_dir)
            self.generate_op_host()
        finally:
            os.chdir(cwd)

        cwd = os.getcwd()
        try:
            os.chdir(op_kernel_dir)
            self.dump_kernel(self.name)
        finally:
            os.chdir(cwd)

        resources_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources"))
        tensorutils_src = os.path.join(resources_dir, "tensorutils.h")
        if not os.path.isfile(tensorutils_src):
            raise FileNotFoundError(f"未找到tensorutils.h: {tensorutils_src}")
        shutil.copy2(tensorutils_src, os.path.join(op_kernel_dir, "tensorutils.h"))
