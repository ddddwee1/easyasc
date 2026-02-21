import logging
import math
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch

from ..utils.Tensor import DBuff, GMTensor, Tensor
from ..utils.datatype import DataTypeValue
from ..utils.pipe import Pipe, PipeType
from ..utils.var import Expr, Var
from ._core_utils import validate_core_idx
from .pipe import FIXPipe, MPipe, MTE1Pipe, MTE2Pipe, SimInstruction

if TYPE_CHECKING:
    from ..utils.instruction import Instruction


Number = Union[int, float, bool]
_SIM_LOGGER = logging.getLogger("easyasc.simulator.cube")
if not _SIM_LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _SIM_LOGGER.addHandler(_handler)
_SIM_LOGGER.setLevel(logging.INFO)
_SIM_LOGGER.propagate = False


class Cube:
    def __init__(
        self,
        core_idx: int,
        l1: torch.Tensor,
        l0a: torch.Tensor,
        l0b: torch.Tensor,
        l0c: torch.Tensor,
        ub1: torch.Tensor,
        ub2: torch.Tensor,
    ) -> None:
        self.core_idx = validate_core_idx(core_idx)

        self.MTE2 = MTE2Pipe(core_idx)
        self.MTE1 = MTE1Pipe(core_idx)
        self.M = MPipe(core_idx)
        self.FIX = FIXPipe(core_idx)

        self.L1 = l1
        self.L0A = l0a
        self.L0B = l0b
        self.L0C = l0c
        self.UB1 = ub1
        self.UB2 = ub2

        self.var_values: Dict[str, Number] = {}
        self.buffer_views: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = {}
        self.gm_views: Dict[str, Optional[torch.Tensor]] = {}
        self._alloc_offsets: Dict[str, int] = {
            "L1": 0,
            "L0A": 0,
            "L0B": 0,
            "L0C": 0,
            "UB": 0,
        }
        self.atomic_enabled = False
        self.atomic_dtype: Optional[DataTypeValue] = None
        self._dispatch_seq = 0
        self._sync_tokens: Dict[Tuple[str, str, str, str], bool] = {}

    def run(
        self,
        instructions: List["Instruction"],
        bound_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.var_values = {}
        self.buffer_views = {}
        self.gm_views = {}
        self.atomic_enabled = False
        self.atomic_dtype = None
        self._dispatch_seq = 0
        self._sync_tokens = {}
        self._clear_pipes()
        self._seed_bound_args(bound_args)
        self._seed_var_values(instructions)
        self._execute_block(instructions, 0, len(instructions))
        self._execute_pipes()

    def _clear_pipes(self) -> None:
        self.MTE2.clear()
        self.MTE1.clear()
        self.M.clear()
        self.FIX.clear()

    @staticmethod
    def _flag_sync_key(src: str, dst: str, event_id: Any) -> Tuple[str, str, str, str]:
        return ("flag", src, dst, str(event_id))

    @staticmethod
    def _event_sync_key(event_info: Dict[str, Any]) -> Tuple[str, str, str, str]:
        src_pipe = str(event_info.get("src_pipe", ""))
        dst_pipe = str(event_info.get("dst_pipe", ""))
        name = str(event_info.get("name", ""))
        return ("event", src_pipe, dst_pipe, name)

    def _event_to_info(self, event: Any) -> Dict[str, Any]:
        if isinstance(event, dict):
            name = event.get("name", None)
            src_pipe = self._pipe_name(event.get("src_pipe", None))
            dst_pipe = self._pipe_name(event.get("dst_pipe", None))
            preset = bool(event.get("preset", False))
        else:
            name = getattr(event, "name", None)
            src_pipe = self._pipe_name(getattr(event, "src_pipe", None))
            dst_pipe = self._pipe_name(getattr(event, "dst_pipe", None))
            preset = bool(getattr(event, "preset", False))
        if not isinstance(name, str) or name == "":
            raise TypeError(f"event must have non-empty string name, got: {name}")
        if src_pipe == "" or dst_pipe == "":
            raise TypeError(f"event must have valid src/dst pipes, got: src={src_pipe}, dst={dst_pipe}")
        return {
            "name": name,
            "src_pipe": src_pipe,
            "dst_pipe": dst_pipe,
            "preset": preset,
        }

    def _instruction_blocked(self, inst: SimInstruction) -> bool:
        if inst.opname == "waitflag":
            key = self._flag_sync_key(
                str(inst.args.get("src", "")),
                str(inst.args.get("dst", "")),
                inst.args.get("event_id", None),
            )
            return not self._sync_tokens.get(key, False)
        if inst.opname == "event_wait":
            event_info = inst.args.get("event", None)
            if not isinstance(event_info, dict):
                raise TypeError(f"event_wait requires event dict in args, got: {type(event_info)}")
            key = self._event_sync_key(event_info)
            return not self._sync_tokens.get(key, False)
        return False

    def _execute_sync_instruction(self, inst: SimInstruction) -> bool:
        if inst.opname == "setflag":
            key = self._flag_sync_key(
                str(inst.args.get("src", "")),
                str(inst.args.get("dst", "")),
                inst.args.get("event_id", None),
            )
            self._sync_tokens[key] = True
            return True
        if inst.opname == "waitflag":
            key = self._flag_sync_key(
                str(inst.args.get("src", "")),
                str(inst.args.get("dst", "")),
                inst.args.get("event_id", None),
            )
            self._sync_tokens[key] = False
            return True
        if inst.opname in ("event_set", "event_setall"):
            event_info = inst.args.get("event", None)
            if not isinstance(event_info, dict):
                raise TypeError(f"{inst.opname} requires event dict in args, got: {type(event_info)}")
            self._sync_tokens[self._event_sync_key(event_info)] = True
            return True
        if inst.opname in ("event_wait", "event_release"):
            event_info = inst.args.get("event", None)
            if not isinstance(event_info, dict):
                raise TypeError(f"{inst.opname} requires event dict in args, got: {type(event_info)}")
            self._sync_tokens[self._event_sync_key(event_info)] = False
            return True
        return False

    def _execute_pipes(self) -> None:
        pipes = [self.MTE2, self.MTE1, self.M, self.FIX]
        next_indices = {id(pipe): 0 for pipe in pipes}
        total = sum(len(pipe.instructions) for pipe in pipes)
        done = 0
        while done < total:
            candidates: List[Tuple[int, Union[MTE2Pipe, MTE1Pipe, MPipe, FIXPipe], SimInstruction]] = []
            for pipe in pipes:
                idx = next_indices[id(pipe)]
                if idx >= len(pipe.instructions):
                    continue
                inst = pipe.instructions[idx]
                if self._instruction_blocked(inst):
                    continue
                candidates.append((inst.seq, pipe, inst))
            if not candidates:
                blocked: List[str] = []
                for pipe in pipes:
                    idx = next_indices[id(pipe)]
                    if idx >= len(pipe.instructions):
                        continue
                    inst = pipe.instructions[idx]
                    blocked.append(f"{pipe.pipe_name}:{inst.opname}")
                raise RuntimeError(
                    "Simulator deadlock while executing pipes with auto-sync waits: "
                    + ", ".join(blocked)
                )
            _, pipe, inst = min(candidates, key=lambda item: item[0])
            if not self._execute_sync_instruction(inst):
                pipe.execute_instruction(inst)
            next_indices[id(pipe)] += 1
            done += 1

    def _seed_var_values(self, instructions: List["Instruction"]) -> None:
        def _visit(value: Any) -> None:
            if isinstance(value, Var):
                if isinstance(value.value, (int, float, bool)):
                    self.var_values.setdefault(value.name, value.value)
                return
            if isinstance(value, dict):
                for item in value.values():
                    _visit(item)
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    _visit(item)
                return

        for inst in instructions:
            for value in inst.kwargs.values():
                _visit(value)

    def _seed_bound_args(self, bound_args: Optional[Dict[str, Any]]) -> None:
        if not isinstance(bound_args, dict):
            return
        for key, value in bound_args.items():
            if not isinstance(key, str):
                continue
            if not isinstance(value, Var):
                continue
            if not isinstance(value.value, (int, float, bool)):
                continue
            self.var_values.setdefault(key, value.value)
            self.var_values.setdefault(value.name, value.value)

    @staticmethod
    def _dtype_size(dtype: DataTypeValue) -> int:
        try:
            return int(dtype.size)
        except Exception:
            if str(dtype) == "bfloat16_t":
                return 2
            raise

    @staticmethod
    def _to_torch_dtype(dtype: DataTypeValue) -> torch.dtype:
        name = str(dtype)
        mapping: Dict[str, torch.dtype] = {
            "half": torch.float16,
            "float": torch.float32,
            "int": torch.int32,
            "int8_t": torch.int8,
            "uint8_t": torch.uint8,
            "int16_t": torch.int16,
            "int64_t": torch.int64,
            "bfloat16_t": torch.bfloat16,
            "bfloat16": torch.bfloat16,
        }
        if hasattr(torch, "uint16"):
            mapping["uint16_t"] = torch.uint16  # type: ignore[attr-defined]
        if hasattr(torch, "uint32"):
            mapping["uint32_t"] = torch.uint32  # type: ignore[attr-defined]
        if hasattr(torch, "uint64"):
            mapping["uint64_t"] = torch.uint64  # type: ignore[attr-defined]
        if name not in mapping:
            raise ValueError(f"Unsupported dtype for simulator tensor view: {dtype}")
        return mapping[name]

    def _resolve_scalar(self, value: Any) -> Number:
        if isinstance(value, Var):
            if value.name in self.var_values:
                return self.var_values[value.name]
            if isinstance(value.value, (int, float, bool)):
                return value.value
            raise ValueError(f"Var {value.name} has no runtime value in simulator")
        if isinstance(value, Expr):
            return self._eval_expr(str(value))
        if isinstance(value, (int, float, bool)):
            return value
        raise TypeError(f"Unsupported scalar value type: {type(value)}")

    def _resolve_int(self, value: Any, label: str) -> int:
        scalar = self._resolve_scalar(value)
        if isinstance(scalar, bool):
            return int(scalar)
        if not isinstance(scalar, (int, float)):
            raise TypeError(f"{label} must resolve to int/float, got: {type(scalar)}")
        return int(scalar)

    def _resolve_bool(self, value: Any) -> bool:
        if isinstance(value, Expr):
            return bool(self._eval_expr(str(value)))
        if isinstance(value, Var):
            return bool(self._resolve_scalar(value))
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        raise TypeError(f"Unsupported condition type: {type(value)}")

    def _eval_expr(self, expr: str) -> Number:
        py_expr = expr.replace("&&", " and ").replace("||", " or ")
        py_expr = re.sub(r"(?<![=!<>])!(?!=)", " not ", py_expr)
        py_expr = re.sub(r"\btrue\b", "True", py_expr, flags=re.IGNORECASE)
        py_expr = re.sub(r"\bfalse\b", "False", py_expr, flags=re.IGNORECASE)
        env: Dict[str, Number] = dict(self.var_values)
        try:
            out = eval(py_expr, {"__builtins__": {}}, env)
        except Exception as exc:
            raise ValueError(f"Failed to evaluate expression {expr!r}: {exc}") from exc
        if not isinstance(out, (int, float, bool)):
            raise TypeError(f"Expression result must be int/float/bool, got: {type(out)}")
        return out

    def _get_memory_by_position(self, position: str) -> torch.Tensor:
        if position == "L1":
            return self.L1
        if position == "L0A":
            return self.L0A
        if position == "L0B":
            return self.L0B
        if position == "L0C":
            return self.L0C
        if position == "UB":
            return self.UB1
        raise ValueError(f"Unsupported local position for cube simulator: {position}")

    def _alloc_bytes(self, position: str, size_bytes: int) -> torch.Tensor:
        if size_bytes < 0:
            raise ValueError(f"size_bytes must be non-negative, got: {size_bytes}")
        memory = self._get_memory_by_position(position)
        start = self._alloc_offsets[position]
        end = start + size_bytes
        if end > int(memory.numel()):
            raise MemoryError(
                f"{position} memory overflow: request={size_bytes} bytes, "
                f"offset={start}, capacity={int(memory.numel())}"
            )
        self._alloc_offsets[position] = end
        return memory[start:end]

    def _allocate_tensor(self, tensor: Tensor) -> torch.Tensor:
        position = str(tensor.position)
        shape0 = self._resolve_int(tensor.shape[0], "shape[0]")
        shape1 = self._resolve_int(tensor.shape[1], "shape[1]")
        numel = shape0 * shape1
        dtype = tensor.dtype
        elem_size = self._dtype_size(dtype)
        raw = self._alloc_bytes(position, numel * elem_size)
        torch_dtype = self._to_torch_dtype(dtype)
        return raw.view(torch_dtype).view(shape0, shape1)

    def _allocate_dbuf(self, dbuf: DBuff) -> Tuple[torch.Tensor, torch.Tensor]:
        position = str(dbuf.position)
        shape0 = self._resolve_int(dbuf.shape[0], "shape[0]")
        shape1 = self._resolve_int(dbuf.shape[1], "shape[1]")
        numel = shape0 * shape1
        dtype = dbuf.dtype
        elem_size = self._dtype_size(dtype)
        torch_dtype = self._to_torch_dtype(dtype)
        first = self._alloc_bytes(position, numel * elem_size).view(torch_dtype).view(shape0, shape1)
        second = self._alloc_bytes(position, numel * elem_size).view(torch_dtype).view(shape0, shape1)
        return first, second

    def _handle_create_var(self, inst: "Instruction") -> None:
        val = inst.kwargs.get("val")
        if not isinstance(val, Var):
            raise TypeError(f"create_var requires Var value, got: {type(val)}")
        raw_value = val.value
        if isinstance(raw_value, (int, float, bool)):
            self.var_values[val.name] = raw_value
        elif raw_value is None:
            self.var_values[val.name] = 0
        else:
            self.var_values[val.name] = self._resolve_scalar(raw_value)

    def _handle_create_tensor(self, inst: "Instruction") -> None:
        val = inst.kwargs.get("val")
        if not isinstance(val, Tensor):
            raise TypeError(f"create_tensor requires Tensor value, got: {type(val)}")
        self.buffer_views[val.name] = self._allocate_tensor(val)

    def _handle_create_dbuf(self, inst: "Instruction") -> None:
        val = inst.kwargs.get("val")
        if not isinstance(val, DBuff):
            raise TypeError(f"create_dbuf requires DBuff value, got: {type(val)}")
        self.buffer_views[val.name] = self._allocate_dbuf(val)

    def _get_tensor_view(self, tensor: Tensor) -> torch.Tensor:
        view = self.buffer_views.get(tensor.name)
        if isinstance(view, tuple):
            raise TypeError(f"Tensor {tensor.name} unexpectedly mapped to DBuff tuple")
        if view is None:
            raise KeyError(f"Tensor view not found for {tensor.name}")
        return view

    def _get_gm_view(self, tensor: GMTensor) -> torch.Tensor:
        view = self.gm_views.get(tensor.name)
        if view is None:
            raise ValueError(
                f"GMTensor data is not available for {tensor.name}; "
                "please provide input data for simulator execution"
            )
        return view

    def _handle_get_buf(self, inst: "Instruction") -> None:
        buf = inst.kwargs.get("buf")
        index = inst.kwargs.get("index")
        out = inst.kwargs.get("out")
        if not isinstance(buf, DBuff):
            raise TypeError(f"get_buf requires DBuff buf, got: {type(buf)}")
        if not isinstance(out, Tensor):
            raise TypeError(f"get_buf requires Tensor out, got: {type(out)}")
        views = self.buffer_views.get(buf.name)
        if not isinstance(views, tuple):
            raise KeyError(f"DBuff views not found for {buf.name}")
        idx = self._resolve_int(index, "get_buf index")
        self.buffer_views[out.name] = views[idx % 2]

    def _handle_slice_tensor(self, inst: "Instruction") -> None:
        src = inst.kwargs.get("src")
        out = inst.kwargs.get("out")
        offset = inst.kwargs.get("offset")
        span = inst.kwargs.get("span")
        step = inst.kwargs.get("step")
        if not isinstance(src, Tensor):
            raise TypeError(f"slice_tensor requires Tensor src, got: {type(src)}")
        if not isinstance(out, Tensor):
            raise TypeError(f"slice_tensor requires Tensor out, got: {type(out)}")
        if not isinstance(offset, (list, tuple)) or not isinstance(span, (list, tuple)):
            raise TypeError("slice_tensor requires list/tuple offset/span")
        src_view = self._get_tensor_view(src)
        row0 = self._resolve_int(offset[0], "slice_tensor offset[0]")
        col0 = self._resolve_int(offset[1], "slice_tensor offset[1]")
        row_span = self._resolve_int(span[0], "slice_tensor span[0]")
        col_span = self._resolve_int(span[1], "slice_tensor span[1]")
        row_step = 1
        col_step = 1
        if isinstance(step, (list, tuple)) and len(step) >= 2:
            row_step = self._resolve_int(step[0], "slice_tensor step[0]")
            col_step = self._resolve_int(step[1], "slice_tensor step[1]")
        self.buffer_views[out.name] = src_view[
            row0 : row0 + row_span : row_step,
            col0 : col0 + col_span : col_step,
        ]

    def _handle_create_gm_tensor(self, inst: "Instruction") -> None:
        val = inst.kwargs.get("val")
        if not isinstance(val, GMTensor):
            raise TypeError(f"create_gm_tensor requires GMTensor value, got: {type(val)}")
        self.gm_views[val.name] = val.data

    def _handle_slice_gm_tensor(self, inst: "Instruction") -> None:
        src = inst.kwargs.get("src")
        out = inst.kwargs.get("out")
        offset = inst.kwargs.get("offset")
        span = inst.kwargs.get("span")
        step = inst.kwargs.get("step")
        if not isinstance(src, GMTensor):
            raise TypeError(f"slice_gm_tensor requires GMTensor src, got: {type(src)}")
        if not isinstance(out, GMTensor):
            raise TypeError(f"slice_gm_tensor requires GMTensor out, got: {type(out)}")
        if not isinstance(offset, (list, tuple)) or not isinstance(span, (list, tuple)):
            raise TypeError("slice_gm_tensor requires list/tuple offset/span")
        src_view = self._get_gm_view(src)
        row0 = self._resolve_int(offset[0], "slice_gm_tensor offset[0]")
        col0 = self._resolve_int(offset[1], "slice_gm_tensor offset[1]")
        row_span = self._resolve_int(span[0], "slice_gm_tensor span[0]")
        col_span = self._resolve_int(span[1], "slice_gm_tensor span[1]")
        row_step = 1
        col_step = 1
        if isinstance(step, (list, tuple)) and len(step) >= 2:
            row_step = self._resolve_int(step[0], "slice_gm_tensor step[0]")
            col_step = self._resolve_int(step[1], "slice_gm_tensor step[1]")
        self.gm_views[out.name] = src_view[
            row0 : row0 + row_span : row_step,
            col0 : col0 + col_span : col_step,
        ]

    def _cube_num(self) -> int:
        from .. import globvars

        device_type = str(getattr(globvars, "device_type", "")).lower()
        if device_type in ("b3", "b4"):
            return 20
        if device_type in ("b1", "b2"):
            return 24
        if device_type == "950":
            return 32
        raise ValueError(f"Unsupported device_type for simulator: {device_type}")

    def _assign_var_op(self, inst: "Instruction") -> None:
        out = inst.kwargs.get("out")
        if not isinstance(out, Var):
            raise TypeError(f"{inst.opname} requires Var out, got: {type(out)}")
        opname = inst.opname
        if opname == "GetCubeNum":
            value: Number = self._cube_num()
        elif opname == "GetCubeIdx":
            value = self.core_idx
        elif opname == "GetVecNum":
            value = self._cube_num() * 2
        elif opname == "GetVecIdx":
            value = self.core_idx * 2
        elif opname == "GetSubBlockIdx":
            value = 0
        elif opname == "CeilDiv":
            a = self._resolve_int(inst.kwargs.get("a"), "CeilDiv a")
            b = self._resolve_int(inst.kwargs.get("b"), "CeilDiv b")
            value = (a + b - 1) // b
        elif opname == "Min":
            a = self._resolve_scalar(inst.kwargs.get("a"))
            b = self._resolve_scalar(inst.kwargs.get("b"))
            value = min(a, b)
        elif opname == "Max":
            a = self._resolve_scalar(inst.kwargs.get("a"))
            b = self._resolve_scalar(inst.kwargs.get("b"))
            value = max(a, b)
        elif opname == "var_mul":
            a = self._resolve_scalar(inst.kwargs.get("a"))
            b = self._resolve_scalar(inst.kwargs.get("b"))
            value = a * b
        elif opname == "var_div":
            a = self._resolve_scalar(inst.kwargs.get("a"))
            b = self._resolve_scalar(inst.kwargs.get("b"))
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                raise TypeError("var_div operands must resolve to numeric values")
            if b == 0:
                raise ZeroDivisionError("var_div divide by zero")
            if str(getattr(out, "dtype", "")) == "int":
                value = int(a) // int(b)
            else:
                value = a / b
        elif opname == "var_add":
            a = self._resolve_scalar(inst.kwargs.get("a"))
            b = self._resolve_scalar(inst.kwargs.get("b"))
            value = a + b
        elif opname == "var_sub":
            a = self._resolve_scalar(inst.kwargs.get("a"))
            b = self._resolve_scalar(inst.kwargs.get("b"))
            value = a - b
        elif opname == "scalar_sqrt":
            a = self._resolve_scalar(inst.kwargs.get("a"))
            if not isinstance(a, (int, float)):
                raise TypeError(f"scalar_sqrt requires numeric operand, got: {type(a)}")
            value = math.sqrt(float(a))
        elif opname in ("Align16", "Align32", "Align64", "Align128", "Align256"):
            a = self._resolve_int(inst.kwargs.get("a"), opname + " a")
            align = int(opname.replace("Align", ""))
            value = ((a + align - 1) // align) * align
        else:
            raise ValueError(f"Unsupported scalar assignment op in cube simulator: {opname}")
        self.var_values[out.name] = value

    @staticmethod
    def _pipe_name(pipe: Any) -> str:
        if isinstance(pipe, PipeType):
            return str(pipe)
        if isinstance(pipe, str):
            return pipe
        return ""

    def _pipe_from_name(self, pipe_name: str) -> Optional[Union[MTE2Pipe, MTE1Pipe, MPipe, FIXPipe]]:
        if pipe_name == "MTE2":
            return self.MTE2
        if pipe_name == "MTE1":
            return self.MTE1
        if pipe_name == "M":
            return self.M
        if pipe_name == "FIX":
            return self.FIX
        return None

    def _resolve_pipe(self, inst: "Instruction") -> Optional[Union[MTE2Pipe, MTE1Pipe, MPipe, FIXPipe]]:
        opname = inst.opname
        if opname == "gm_to_l1_nd2nz":
            return self.MTE2
        if opname == "l1_to_l0":
            return self.MTE1
        if opname == "mmad":
            return self.M
        if opname in ("l0c_to_gm_nz2nd", "l0c_to_l1"):
            return self.FIX
        if opname == "sim_print":
            return self._pipe_from_name(self._pipe_name(inst.kwargs.get("pipe")))
        if opname == "setflag":
            return self._pipe_from_name(self._pipe_name(inst.kwargs.get("src")))
        if opname == "waitflag":
            return self._pipe_from_name(self._pipe_name(inst.kwargs.get("dst")))
        if opname in ("event_set", "event_setall", "event_release"):
            event_info = self._event_to_info(inst.kwargs.get("event", None))
            return self._pipe_from_name(str(event_info["src_pipe"]))
        if opname == "event_wait":
            event_info = self._event_to_info(inst.kwargs.get("event", None))
            return self._pipe_from_name(str(event_info["dst_pipe"]))
        if opname in ("cube_ready", "wait_vec", "allcube_ready", "allcube_wait", "barrier"):
            return self._pipe_from_name(self._pipe_name(inst.kwargs.get("pipe")))
        return None

    def _resolve_misc_value(self, value: Any) -> Any:
        if isinstance(value, Var):
            return self._resolve_scalar(value)
        if isinstance(value, Expr):
            return self._eval_expr(str(value))
        if isinstance(value, PipeType):
            return str(value)
        if isinstance(value, (int, float, bool, str)):
            return value
        if isinstance(value, (list, tuple)):
            out: List[Any] = []
            for item in value:
                out.append(self._resolve_misc_value(item))
            if isinstance(value, tuple):
                return tuple(out)
            return out
        if value is None:
            return None
        return str(value)

    def _build_sim_instruction(self, inst: "Instruction") -> SimInstruction:
        args: Dict[str, Any] = {}
        tensors: Dict[str, Any] = {}
        tensor_dtypes: Dict[str, str] = {}
        for key, value in inst.kwargs.items():
            if inst.opname == "sim_print" and key == "payload":
                if not isinstance(value, (list, tuple)):
                    raise TypeError(f"sim_print payload must be list/tuple, got: {type(value)}")
                args[key] = [self._resolve_print_item(item) for item in value]
                continue
            if key == "event":
                args[key] = self._event_to_info(value)
                continue
            if isinstance(value, Tensor):
                tensors[key] = self._get_tensor_view(value)
                tensor_dtypes[key] = str(value.dtype)
                if inst.opname == "l1_to_l0" and key == "src":
                    args["src_is_transpose"] = bool(getattr(value, "is_transpose", False))
                continue
            if isinstance(value, GMTensor):
                tensors[key] = self._get_gm_view(value)
                tensor_dtypes[key] = str(value.dtype)
                continue
            args[key] = self._resolve_misc_value(value)
        atomic_enabled: Optional[bool] = None
        atomic_dtype: Optional[str] = None
        if inst.opname == "l0c_to_gm_nz2nd":
            atomic_enabled = self.atomic_enabled
            atomic_dtype = str(self.atomic_dtype) if self.atomic_dtype is not None else None
        seq = self._dispatch_seq
        self._dispatch_seq += 1
        return SimInstruction(
            opname=inst.opname,
            args=args,
            tensors=tensors,
            tensor_dtypes=tensor_dtypes,
            atomic_enabled=atomic_enabled,
            atomic_dtype=atomic_dtype,
            seq=seq,
        )

    def _resolve_print_item(self, item: Any) -> Any:
        if isinstance(item, Tensor):
            return self._get_tensor_view(item)
        if isinstance(item, GMTensor):
            return self._get_gm_view(item)
        if isinstance(item, (Var, Expr, PipeType, int, float, bool, str, list, tuple)):
            return self._resolve_misc_value(item)
        return item

    def _log_sim_print(self, pipe_name: str, payload: List[Any]) -> None:
        message = " ".join(str(item) for item in payload)
        prefix = f"[cube][core={self.core_idx}]"
        if pipe_name != str(Pipe.S):
            prefix = f"{prefix}[pipe={pipe_name}]"
        _SIM_LOGGER.info(f"{prefix} {message}")

    def _dispatch_to_pipe(self, inst: "Instruction") -> bool:
        if inst.opname == "barrier" and self._pipe_name(inst.kwargs.get("pipe")) == "ALL":
            for pipe in (self.MTE2, self.MTE1, self.M, self.FIX):
                pipe.issue(self._build_sim_instruction(inst))
            return True
        pipe = self._resolve_pipe(inst)
        if pipe is None:
            return False
        pipe.issue(self._build_sim_instruction(inst))
        return True

    def _find_loop_end(self, instructions: List["Instruction"], start_idx: int, end_idx: int) -> int:
        depth = 1
        idx = start_idx + 1
        while idx < end_idx:
            opname = instructions[idx].opname
            if opname == "start_loop":
                depth += 1
            elif opname == "end_loop":
                depth -= 1
                if depth == 0:
                    return idx
            idx += 1
        raise ValueError("start_loop without matching end_loop")

    def _find_if_end(self, instructions: List["Instruction"], start_idx: int, end_idx: int) -> int:
        depth = 1
        idx = start_idx + 1
        while idx < end_idx:
            opname = instructions[idx].opname
            if opname == "start_if":
                depth += 1
            elif opname == "end_if":
                depth -= 1
                if depth == 0:
                    return idx
            idx += 1
        raise ValueError("start_if without matching end_if")

    def _execute_if_chain(self, instructions: List["Instruction"], start_idx: int, end_idx: int) -> int:
        if_end = self._find_if_end(instructions, start_idx, end_idx)
        segments: List[Tuple[int, int, int]] = []
        segment_header = start_idx
        segment_body_start = start_idx + 1
        nested_if = 0
        nested_loop = 0
        idx = start_idx + 1
        while idx < if_end:
            opname = instructions[idx].opname
            if opname == "start_loop":
                nested_loop += 1
            elif opname == "end_loop":
                nested_loop -= 1
            elif opname == "start_if":
                nested_if += 1
            elif opname == "end_if":
                nested_if -= 1
            elif opname in ("start_elif", "start_else") and nested_if == 0 and nested_loop == 0:
                segments.append((segment_header, segment_body_start, idx))
                segment_header = idx
                segment_body_start = idx + 1
            idx += 1
        segments.append((segment_header, segment_body_start, if_end))

        for header_idx, body_start, body_end in segments:
            header_inst = instructions[header_idx]
            if header_inst.opname == "start_else":
                self._execute_block(instructions, body_start, body_end)
                break
            cond = header_inst.kwargs.get("cond")
            if self._resolve_bool(cond):
                self._execute_block(instructions, body_start, body_end)
                break
        return if_end + 1

    def _execute_inst(self, inst: "Instruction") -> None:
        opname = inst.opname
        if opname == "reset_cache":
            self._alloc_offsets = {key: 0 for key in self._alloc_offsets}
            return
        if opname == "create_var":
            self._handle_create_var(inst)
            return
        if opname in (
            "GetCubeNum",
            "GetCubeIdx",
            "GetVecNum",
            "GetVecIdx",
            "GetSubBlockIdx",
            "CeilDiv",
            "Min",
            "Max",
            "var_mul",
            "var_div",
            "var_add",
            "var_sub",
            "scalar_sqrt",
            "Align16",
            "Align32",
            "Align64",
            "Align128",
            "Align256",
        ):
            self._assign_var_op(inst)
            return
        if opname == "create_tensor":
            self._handle_create_tensor(inst)
            return
        if opname == "create_dbuf":
            self._handle_create_dbuf(inst)
            return
        if opname == "get_buf":
            self._handle_get_buf(inst)
            return
        if opname == "slice_tensor":
            self._handle_slice_tensor(inst)
            return
        if opname == "create_gm_tensor":
            self._handle_create_gm_tensor(inst)
            return
        if opname == "slice_gm_tensor":
            self._handle_slice_gm_tensor(inst)
            return
        if opname in ("create_sevent", "create_devent"):
            event_info = self._event_to_info(inst.kwargs.get("val", None))
            self._sync_tokens[self._event_sync_key(event_info)] = bool(event_info.get("preset", False))
            return
        if opname == "gm_to_l1_nd2nz":
            self._dispatch_to_pipe(inst)
            return
        if opname in ("atomic_add", "atomic_max", "atomic_min"):
            self.atomic_enabled = True
            return
        if opname == "set_atomic_type":
            dtype = inst.kwargs.get("dtype")
            if isinstance(dtype, DataTypeValue):
                self.atomic_dtype = dtype
                self.atomic_enabled = True
            return
        if opname == "atomic_end":
            self.atomic_enabled = False
            self.atomic_dtype = None
            return
        if opname == "sim_print":
            pipe_name = self._pipe_name(inst.kwargs.get("pipe"))
            if pipe_name == "":
                pipe_name = str(Pipe.S)
            if pipe_name == str(Pipe.S):
                raw_payload = inst.kwargs.get("payload", [])
                if not isinstance(raw_payload, (list, tuple)):
                    raise TypeError(f"sim_print payload must be list/tuple, got: {type(raw_payload)}")
                payload = [self._resolve_print_item(item) for item in raw_payload]
                self._log_sim_print(pipe_name, payload)
            else:
                self._dispatch_to_pipe(inst)
            return
        if opname in ("start_auto_sync", "end_auto_sync"):
            return
        self._dispatch_to_pipe(inst)

    def _execute_block(self, instructions: List["Instruction"], start_idx: int, end_idx: int) -> None:
        idx = start_idx
        while idx < end_idx:
            inst = instructions[idx]
            opname = inst.opname
            if opname == "start_loop":
                loop_end = self._find_loop_end(instructions, idx, end_idx)
                loop_var = inst.kwargs.get("var")
                if not isinstance(loop_var, Var):
                    raise TypeError(f"start_loop requires Var loop variable, got: {type(loop_var)}")
                start = self._resolve_int(inst.kwargs.get("start"), "start_loop start")
                stop = self._resolve_int(inst.kwargs.get("stop"), "start_loop stop")
                step = self._resolve_int(inst.kwargs.get("step"), "start_loop step")
                if step == 0:
                    raise ValueError("start_loop step cannot be zero")
                for value in range(start, stop, step):
                    self.var_values[loop_var.name] = value
                    self._execute_block(instructions, idx + 1, loop_end)
                idx = loop_end + 1
                continue
            if opname == "end_loop":
                return
            if opname == "start_if":
                idx = self._execute_if_chain(instructions, idx, end_idx)
                continue
            if opname in ("start_elif", "start_else", "end_if"):
                return
            self._execute_inst(inst)
            idx += 1
