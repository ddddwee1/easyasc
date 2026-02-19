import inspect
import json
import os
import platform
import shutil
import tarfile
from typing import List, Optional, Set, Union, TYPE_CHECKING

from ..utils.instruction import Instruction
from ..utils.Tensor import GMTensor, DBuff
from ..utils.var import Var
from ..utils.mutex import CvMutex, VcMutex
from ..utils.datatype import Datatype
from ..utils.positions import Position
from .. import globvars

if TYPE_CHECKING:
    from ..micro.micromodule import MicroModule


class KernelBase:
    """Kernel base class that stores name, function, and emitted instructions."""
    def __init__(self, name: str, func):
        self.name = name
        self.func = func
        self.instructions: List[Instruction] = []
        self.crosscore_mutex: List[Union[CvMutex, VcMutex]] = []
        self.workspace_shapes: List[Union[list, tuple]] = []
        self.used_micros: Set["MicroModule"] = set()
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
        self.used_micros = set()
        for param_name, value in bound.arguments.items():
            if isinstance(value, (GMTensor, Var)):
                value.name = param_name
            else:
                raise TypeError(f"kernel arguments must be GMTensor or Var, current {param_name} type: {type(value)}")
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
                    f"crosscore_mutex elements must be CvMutex or VcMutex, got: {type(mutex)}"
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
            header = '#pragma once\n#include "tensorutils.h"\n'
            for i in self.used_micros:
                header += f'#include "{i.name}.h"\n'
            header += '\n'
            fn_name = f"{self.name}_{suffix}"
            body = code.strip("\n")
            if body:
                body = "\n".join(f"    {line}" if line else "" for line in body.splitlines())
                return (
                    f"{header}__aicore__ inline void {fn_name}({param_list}) {{\n"
                    f"    TPipe pipe;\n"
                    f"    TPipe* pipe_ptr = GetTPipePtr();\n"
                    f"    int _offset = 0;\n"
                    f"{body}\n"
                    f"}}\n"
                )
            return (
                f"{header}__aicore__ inline void {fn_name}({param_list}) {{\n"
                f"    TPipe pipe;\n"
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
            raise RuntimeError("generate_op_host requires calling kernel first to bind arguments")

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
            return mapping.get(name, "ge::DT_FLOAT16")   # type: ignore

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
                    raise ValueError(f"workspace_shape contains an unbound Var: {dim.name!r}")
                raise TypeError(f"workspace_shape elements must be int or Var, got: {type(dim)}")

            def _shape_expr(shape) -> str:
                if not isinstance(shape, (list, tuple)):
                    raise TypeError(f"workspace_shape must be list or tuple, got: {type(shape)}")
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
            raise TypeError(f"path must be str, got: {type(path)}")
        if not isinstance(cann_path, str):
            raise TypeError(f"cann_path must be str, got: {type(cann_path)}")
        resources_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources"))
        tar_path = os.path.join(resources_dir, "CustomOp.tar.gz")
        if not os.path.isfile(tar_path):
            raise FileNotFoundError(f"CustomOp archive not found: {tar_path}")
        preset_template = os.path.join(resources_dir, "CMakePresets.json")
        if not os.path.isfile(preset_template):
            raise FileNotFoundError(f"CMakePresets.json template not found: {preset_template}")
        dst = os.path.abspath(path)
        if os.path.exists(dst):
            if not os.path.isdir(dst):
                raise FileExistsError(f"Target path already exists and is not a directory: {dst}")
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
            raise ValueError("ASCEND_CANN_PACKAGE_PATH config not found")
        preset_out = os.path.join(dst, "CMakePresets.json")
        with open(preset_out, "w", encoding="utf-8") as f:
            json.dump(preset_data, f, indent=4)
            f.write("\n")

    @staticmethod
    def _resolve_custom_opp_path(custom_op_path: str) -> str:
        normalized_path = custom_op_path.rstrip("/")
        if normalized_path.endswith("/opp"):
            return normalized_path
        if normalized_path == "":
            return "/opp"
        return f"{normalized_path}/opp"

    def generate(
        self,
        out_dir: str = "",
        cann_path: Optional[str] = None,
        custom_op_path: Optional[str] = None,
        profile: bool = False,
    ) -> None:
        if not isinstance(out_dir, str):
            raise TypeError(f"out_dir must be str, got: {type(out_dir)}")
        if out_dir == "":
            out_dir = self.name
        if not isinstance(profile, bool):
            raise TypeError(f"profile must be bool, got: {type(profile)}")
        if cann_path is None:
            cann_path = os.getenv("ASCEND_HOME_PATH")
            if not cann_path:
                raise ValueError("cann_path is None and ASCEND_HOME_PATH is not set; please specify cann_path manually")
        if not isinstance(cann_path, str):
            raise TypeError(f"cann_path must be str, got: {type(cann_path)}")
        if custom_op_path is None:
            custom_op_path = cann_path
        if not isinstance(custom_op_path, str):
            raise TypeError(f"custom_op_path must be str, got: {type(custom_op_path)}")

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
            for micro in self.used_micros:
                micro.gen_code(f"{micro.name}.h")
            self.dump_kernel(self.name)
        finally:
            os.chdir(cwd)

        resources_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources"))
        tensorutils_src = os.path.join(resources_dir, "tensorutils.h")
        if not os.path.isfile(tensorutils_src):
            raise FileNotFoundError(f"tensorutils.h not found: {tensorutils_src}")
        shutil.copy2(tensorutils_src, os.path.join(op_kernel_dir, "tensorutils.h"))
        self.generate_aclnn_test(
            f"{out_dir}_aclnn_test",
            cann_path=cann_path,
            custom_op_path=custom_op_path,
            profile=profile,
        )
        self.generate_bashfiles(out_dir, cann_path, custom_op_path=custom_op_path)

    def generate_aclnn_test(
        self,
        path: str,
        cann_path: Optional[str] = None,
        custom_op_path: Optional[str] = None,
        profile: bool = False,
    ) -> None:
        if not isinstance(path, str):
            raise TypeError(f"path must be str, got: {type(path)}")
        if cann_path is None:
            cann_path = os.getenv("ASCEND_HOME_PATH")
            if not cann_path:
                raise ValueError("cann_path is None and ASCEND_HOME_PATH is not set; please specify cann_path manually")
        if not isinstance(cann_path, str):
            raise TypeError(f"cann_path must be str, got: {type(cann_path)}")
        if custom_op_path is None:
            custom_op_path = cann_path
        if not isinstance(custom_op_path, str):
            raise TypeError(f"custom_op_path must be str, got: {type(custom_op_path)}")
        if not isinstance(profile, bool):
            raise TypeError(f"profile must be bool, got: {type(profile)}")
        resolved_custom_op_path = self._resolve_custom_opp_path(custom_op_path)

        abs_path = os.path.abspath(path)
        os.makedirs(abs_path, exist_ok=True)
        os.makedirs(os.path.join(abs_path, "input"), exist_ok=True)
        os.makedirs(os.path.join(abs_path, "output"), exist_ok=True)
        resources_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources"))
        macros_src = os.path.join(resources_dir, "macros.h")
        if not os.path.isfile(macros_src):
            raise FileNotFoundError(f"macros.h not found: {macros_src}")
        shutil.copy2(macros_src, os.path.join(abs_path, "macros.h"))
        parse_prof_src = os.path.join(resources_dir, "parse_prof.py")
        if not os.path.isfile(parse_prof_src):
            raise FileNotFoundError(f"parse_prof.py not found: {parse_prof_src}")
        shutil.copy2(parse_prof_src, os.path.join(abs_path, "parse_prof.py"))
        setup_aclnn_src = os.path.join(resources_dir, "setup_aclnn.py")
        if not os.path.isfile(setup_aclnn_src):
            raise FileNotFoundError(f"setup_aclnn.py not found: {setup_aclnn_src}")
        with open(setup_aclnn_src, "r", encoding="utf-8") as f:
            setup_lines = f.readlines()
        cann_path_literal = repr(cann_path)
        custom_op_path_literal = repr(resolved_custom_op_path)
        replaced_cann_path = False
        replaced_custom_op_path = False
        for idx, line in enumerate(setup_lines):
            stripped = line.lstrip()
            if stripped.startswith("ascend_toolkit_install_path="):
                setup_lines[idx] = f"ascend_toolkit_install_path={cann_path_literal}\n"
                replaced_cann_path = True
                continue
            if stripped.startswith("custom_op_path=") or stripped.startswith("custom_op_path ="):
                setup_lines[idx] = f"custom_op_path = {custom_op_path_literal}\n"
                replaced_custom_op_path = True
        if not replaced_cann_path:
            raise ValueError("ascend_toolkit_install_path config not found in setup_aclnn.py")
        if not replaced_custom_op_path:
            raise ValueError("custom_op_path config not found in setup_aclnn.py")
        machine = platform.machine().lower()
        if machine in ("x86_64", "amd64"):
            arch_tag = "x86_64"
        elif machine in ("aarch64", "arm64"):
            arch_tag = "aarch64"
        else:
            raise ValueError(f"Unknown system architecture: {machine}")
        arch_token = f"{arch_tag}-linux"
        setup_lines = [
            line.replace("aarch64-linux", arch_token).replace("x86_64-linux", arch_token)
            for line in setup_lines
        ]
        with open(os.path.join(abs_path, "setup_aclnn.py"), "w", encoding="utf-8") as f:
            f.writelines(setup_lines)
        tensorx_src = os.path.join(resources_dir, "tensorx.h")
        if not os.path.isfile(tensorx_src):
            raise FileNotFoundError(f"tensorx.h not found: {tensorx_src}")
        with open(tensorx_src, "r", encoding="utf-8") as f:
            tensorx_lines = f.readlines()
        include_replaced = False
        for idx, line in enumerate(tensorx_lines):
            if line.lstrip().startswith('#include "cust_op_list.h"'):
                tensorx_lines[idx] = f'#include "aclnn_{self.name}.h"\n'
                include_replaced = True
                break
        if not include_replaced:
            raise ValueError('Missing #include "cust_op_list.h" in tensorx.h')
        with open(os.path.join(abs_path, "tensorx.h"), "w", encoding="utf-8") as f:
            f.writelines(tensorx_lines)
        if not self._last_bound_args:
            raise RuntimeError("generate_aclnn_test requires calling kernel first to bind arguments")
        sig = inspect.signature(self.func)
        param_names = list(sig.parameters.keys())
        output_gmtensors = set()
        if isinstance(getattr(self, "_last_output_gmtensors", None), set):
            output_gmtensors = self._last_output_gmtensors

        var_params = []
        gmtensor_params = []
        for name in param_names:
            bound_val = self._last_bound_args.get(name, None)
            if isinstance(bound_val, Var):
                var_params.append((name, bound_val))
            elif isinstance(bound_val, GMTensor):
                gmtensor_params.append((name, bound_val))
            else:
                raise TypeError(f"kernel arguments must be GMTensor or Var, current {name} type: {type(bound_val)}")

        gmtensor_values = {val for _, val in gmtensor_params}
        for out in output_gmtensors:
            if out not in gmtensor_values:
                raise ValueError("Output GMTensor must come from kernel arguments")

        var_name_set = {name for name, _ in var_params}

        def _to_camel(name: str) -> str:
            parts = [p for p in name.split("_") if p]
            if not parts:
                return name
            return "".join(p[:1].upper() + p[1:] for p in parts)

        def _var_decl(name: str, var: Var) -> str:
            if var.dtype is None or var.value is None:
                raise ValueError(f"Var {name} has no valid dtype or value")
            if var.dtype is Datatype.int:
                ctype = "int"
            elif var.dtype is Datatype.float:
                ctype = "float"
            else:
                raise ValueError(f"Var {name} only supports int or float dtype, current: {var.dtype}")
            if not isinstance(var.value, (int, float)):
                raise ValueError(f"Var {name} value must be int or float, current: {type(var.value)}")
            return f"    {ctype} {name} = {var.value};"

        def _tensorx_type(dtype) -> str:
            name = getattr(dtype, "name", None)
            mapping = {
                "half": "FP16",
                "float": "FP32",
                "bfloat16_t": "BF16",
                "int": "INT32",
                "int8_t": "INT8",
                "int16_t": "INT16",
                "int64_t": "INT64",
                "uint8_t": "UINT8",
                "uint16_t": "UINT16",
                "uint32_t": "UINT32",
                "uint64_t": "UINT64",
            }
            if name in ("int4", "hif8", "bool"):
                raise ValueError(f"Unsupported dtype: {name}")
            if name not in mapping:
                raise ValueError(f"dtype mapping not found: {name}")
            return mapping[name]

        def _dim_expr(dim) -> str:
            if isinstance(dim, int):
                return str(dim)
            if isinstance(dim, Var):
                if dim.name in var_name_set:
                    if dim.dtype is not Datatype.int:
                        raise ValueError(f"shape contains a non-int Var: {dim.name}")
                    return dim.name
                if isinstance(dim.value, int):
                    return str(dim.value)
                raise ValueError(f"shape contains an unbound Var: {dim.name!r}")
            raise TypeError(f"shape elements must be int or Var, got: {type(dim)}")

        def _shape_expr(shape) -> str:
            if not isinstance(shape, (list, tuple)):
                raise TypeError(f"shape must be list or tuple, got: {type(shape)}")
            return ", ".join(_dim_expr(dim) for dim in shape)

        output_names = {name for name, val in gmtensor_params if val in output_gmtensors}
        input_params = [(name, val) for name, val in gmtensor_params if name not in output_names]
        output_params = [(name, val) for name, val in gmtensor_params if name in output_names]

        profile_pre_lines = []
        profile_post_lines = []
        if profile:
            template_src = os.path.join(resources_dir, "test.cpp")
            if not os.path.isfile(template_src):
                raise FileNotFoundError(f"test.cpp template not found: {template_src}")
            with open(template_src, "r", encoding="utf-8") as f:
                legacy_lines = f.readlines()

            def _extract_comment_block(anchor: str) -> list:
                for i, line in enumerate(legacy_lines):
                    if anchor in line:
                        start = i
                        while start - 1 >= 0 and legacy_lines[start - 1].lstrip().startswith("//"):
                            start -= 1
                        end = i
                        while end + 1 < len(legacy_lines) and legacy_lines[end + 1].lstrip().startswith("//"):
                            end += 1
                        return [ln.rstrip("\n") for ln in legacy_lines[start : end + 1]]
                return []

            profile_pre_lines = _extract_comment_block("aclprofInit")
            profile_post_lines = _extract_comment_block("aclprofStop")
            if not profile_pre_lines or not profile_post_lines:
                raise ValueError("Profiling commented block not found in legacy/test.cpp")
            def _uncomment_line(line: str) -> str:
                prefix, sep, rest = line.partition("//")
                if sep == "":
                    return line
                return f"{prefix}{rest.lstrip()}"
            profile_pre_lines = [_uncomment_line(line) for line in profile_pre_lines]
            profile_post_lines = [_uncomment_line(line) for line in profile_post_lines]

        lines = [
            "#include <iostream>",
            "#include <cstdio>",
            "#include <cstring>",
            "#include <vector>",
            "#include <fstream>",
            "#include <sys/stat.h>",
            '#include "tensorx.h"',
            '#include "acl/acl.h"',
            '#include "acl/acl_prof.h"',
            '#include "macros.h"',
            "",
            "",
            "int main(int argc, char **argv)",
            "{",
            "    int32_t deviceId = 0;",
            "    aclrtContext context;",
            "    aclrtStream stream;",
            "    auto ret = aclInit(nullptr);",
            "    ret = aclrtSetDevice(deviceId);",
            "    ret = aclrtCreateContext(&context, deviceId);",
            "    ret = aclrtSetCurrentContext(context);",
            "    ret = aclrtCreateStream(&stream);",
        ]
        if profile and profile_pre_lines:
            lines.extend(profile_pre_lines)
        lines.extend(
            [
                "    ",
            "    printf(\"--> Initializing tensors...\\n\");",
            ]
        )

        for name, var in var_params:
            lines.append(_var_decl(name, var))
        if var_params:
            lines.append("    ")

        for name, val in input_params:
            tensorx_type = _tensorx_type(val.dtype)
            shape_expr = _shape_expr(val.shape)
            lines.append(f"    TensorX<{tensorx_type}> {name}({{{shape_expr}}});")
            lines.append(f"    {name}.initAll();")
            lines.append(f"    {name}.fillHostWithBinFile(\"./input/input_{name}.bin\");")
            lines.append(f"    {name}.copyToDevice();")
        if input_params:
            lines.append("    ")

        for name, val in output_params:
            tensorx_type = _tensorx_type(val.dtype)
            shape_expr = _shape_expr(val.shape)
            lines.append(f"    TensorX<{tensorx_type}> {name}({{{shape_expr}}});")
            lines.append(f"    {name}.initAll();")
        if output_params:
            lines.append("    ")

        for name, _ in gmtensor_params:
            lines.append(f"    aclTensor* {name}_acl = {name}.toAclTensor();")
        if gmtensor_params:
            lines.append("    ")

        op_name = _to_camel(self.name)
        lines.append("    printf(\"--> Running OP...\\n\");")
        lines.append("    PREPARE_OP();")
        exec_args = []
        for name in param_names:
            bound_val = self._last_bound_args.get(name, None)
            if isinstance(bound_val, GMTensor) and bound_val not in output_gmtensors:
                exec_args.append(f"{name}_acl")
        for name in param_names:
            bound_val = self._last_bound_args.get(name, None)
            if isinstance(bound_val, Var):
                exec_args.append(name)
        for name in param_names:
            bound_val = self._last_bound_args.get(name, None)
            if isinstance(bound_val, GMTensor) and bound_val in output_gmtensors:
                exec_args.append(f"{name}_acl")
        args_str = ", ".join(exec_args)
        if profile:
            lines.append("    for (int i=0; i<100; ++i){")
            lines.append(f"        EXECOP(aclnn{op_name}, stream, {args_str});")
            lines.append("    }")
        lines.append(f"    EXECOP(aclnn{op_name}, stream, {args_str});")
        lines.append("    CHECK_RET(aclrtSynchronizeStream(stream));")
        lines.append("    ")
        lines.append("    printf(\"--> Saving output tensors...\\n\");")
        for name, _ in output_params:
            lines.append(f"    {name}.copyToHost();")
            lines.append(f"    {name}.saveHostToBinFile(\"output/output_{name}.bin\");")
        if profile and profile_post_lines:
            lines.append("    ")
            lines.extend(profile_post_lines)
        lines.append("    ")
        lines.append("    aclFinalize();")
        lines.append("}")

        with open(os.path.join(abs_path, "test.cpp"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def generate_bashfiles(
        self,
        path: str,
        cann_path: str,
        custom_op_path: Optional[str] = None,
    ) -> None:
        if not isinstance(path, str):
            raise TypeError(f"path must be str, got: {type(path)}")
        if not isinstance(cann_path, str):
            raise TypeError(f"cann_path must be str, got: {type(cann_path)}")
        if custom_op_path is None:
            custom_op_path = cann_path
        if not isinstance(custom_op_path, str):
            raise TypeError(f"custom_op_path must be str, got: {type(custom_op_path)}")
        resolved_custom_op_path = self._resolve_custom_opp_path(custom_op_path)
        script_lines = [
            f"cd {path}",
            "bash build.sh",
            "cd build_out",
            'for f in custom_*.run; do bash "$f"; done',
            f"export LD_LIBRARY_PATH={resolved_custom_op_path}/vendors/customize/op_api/lib/:${{LD_LIBRARY_PATH}}",
            f"cd ../../{path}_aclnn_test",
            "python setup_aclnn.py",
            "",
        ]
        with open("b.sh", "w", encoding="utf-8") as f:
            f.write("\n".join(script_lines))
        run_lines = [
            f"export LD_LIBRARY_PATH={resolved_custom_op_path}/vendors/customize/op_api/lib/:${{LD_LIBRARY_PATH}}",
            f"cd {path}_aclnn_test",
            "./aclnn_test",
            "",
        ]
        with open("r.sh", "w", encoding="utf-8") as f:
            f.write("\n".join(run_lines))
