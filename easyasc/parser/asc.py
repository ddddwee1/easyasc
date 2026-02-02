import os
import re
from functools import lru_cache
from typing import Iterable, List, Set, Tuple

from .asc_handlers import build_handlers
from .asc_utils import (
    assignment_expr,
    build_expr_state,
    dtype_to_cpp,
    is_assignment_op,
    is_tmp_var,
    should_skip_inst,
    uses_var_in_operands,
    value_to_cpp,
)
from .asc_pruning import prune_empty_blocks, prune_unused_decls, prune_unused_vars
from .asc_autosync import insert_auto_sync
from .helper import CodeHelper
from ..utils.instruction import Instruction
from ..utils.var import Var

_EVENT_OPS = {
    "create_sevent",
    "create_devent",
    "event_set",
    "event_wait",
    "event_setall",
    "event_release",
}
_CUBE_PIPE_OPS = {"allcube_ready", "allcube_wait", "cube_ready", "wait_vec"}
_VEC_PIPE_OPS = {"allvec_ready", "allvec_wait", "vec_ready", "wait_cube"}
_PIPE_CUBE_NAMES = {"M", "FIX", "MTE1"}
_PIPE_VEC_NAMES = {"V", "MTE3"}
_INSTRUCTION_RE = re.compile(r"Instruction\(\s*['\"]([A-Za-z0-9_]+)['\"]")


def _collect_opnames_from_file(path: str) -> Set[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return set()
    return set(_INSTRUCTION_RE.findall(text))


def _collect_opnames_from_dir(path: str) -> Set[str]:
    names: Set[str] = set()
    for root, _, files in os.walk(path):
        for filename in files:
            if filename.endswith(".py"):
                names |= _collect_opnames_from_file(os.path.join(root, filename))
    return names


@lru_cache(maxsize=1)
def _get_stub_opnames() -> Tuple[Set[str], Set[str]]:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cube_path = os.path.join(base_dir, "stub_functions", "cube.py")
    vec_dir = os.path.join(base_dir, "stub_functions", "vec")
    cube_ops = _collect_opnames_from_file(cube_path)
    vec_ops = _collect_opnames_from_dir(vec_dir)
    handler_map = build_handlers()
    for opname, handler in handler_map.items():
        module = getattr(handler, "__module__", "")
        if ".vec_" in module:
            vec_ops.add(opname)
        elif module.endswith(".cube"):
            cube_ops.add(opname)
    return cube_ops, vec_ops


def _event_sides(event: object) -> Tuple[bool, bool]:
    if event is None:
        return False, False
    pipes = []
    src = getattr(event, "src_pipe", None)
    dst = getattr(event, "dst_pipe", None)
    if src is not None:
        pipes.append(src)
    if dst is not None:
        pipes.append(dst)
    cube_side = any(str(pipe) in _PIPE_CUBE_NAMES for pipe in pipes)
    vec_side = any(str(pipe) in _PIPE_VEC_NAMES for pipe in pipes)
    return cube_side, vec_side


def _classify_inst(inst: Instruction, cube_ops: Set[str], vec_ops: Set[str]) -> str:
    opname = inst.opname
    if opname in _CUBE_PIPE_OPS:
        return "cube"
    if opname in _VEC_PIPE_OPS:
        return "vec"
    if opname in _EVENT_OPS:
        event = inst.kwargs.get("event")
        if event is None:
            event = inst.kwargs.get("val")
        cube_side, vec_side = _event_sides(event)
        if cube_side and not vec_side:
            return "cube"
        if vec_side and not cube_side:
            return "vec"
        return "both"
    if opname in cube_ops and opname not in vec_ops:
        return "cube"
    if opname in vec_ops and opname not in cube_ops:
        return "vec"
    if opname in cube_ops or opname in vec_ops:
        return "both"
    return "both"


def split_instructions(instructions: Iterable[Instruction]) -> Tuple[List[Instruction], List[Instruction]]:
    cube_ops, vec_ops = _get_stub_opnames()
    cube_insts: List[Instruction] = []
    vec_insts: List[Instruction] = []
    for inst in instructions:
        side = _classify_inst(inst, cube_ops, vec_ops)
        if side in ("cube", "both"):
            cube_insts.append(inst)
        if side in ("vec", "both"):
            vec_insts.append(inst)
    def _classify_for_prune(inst: Instruction) -> str:
        return _classify_inst(inst, cube_ops, vec_ops)

    cube_insts = prune_empty_blocks(cube_insts)
    vec_insts = prune_empty_blocks(vec_insts)
    cube_insts = prune_unused_decls(cube_insts)
    vec_insts = prune_unused_decls(vec_insts)
    cube_insts = prune_unused_vars(cube_insts, "cube", _classify_for_prune)
    vec_insts = prune_unused_vars(vec_insts, "vec", _classify_for_prune)
    cube_insts = prune_unused_decls(cube_insts)
    vec_insts = prune_unused_decls(vec_insts)
    cube_insts = prune_empty_blocks(cube_insts)
    vec_insts = prune_empty_blocks(vec_insts)
    return cube_insts, vec_insts


def _next_emit_index(
    instructions: List[Instruction],
    start_idx: int,
    tmp_var_names: Set[str],
    tmp_tensor_names: Set[str],
    tmp_gmtensor_names: Set[str],
) -> int:
    idx = start_idx
    while idx < len(instructions) and should_skip_inst(
        instructions[idx],
        tmp_var_names,
        tmp_tensor_names,
        tmp_gmtensor_names,
    ):
        idx += 1
    return idx


def _try_fold_loop(
    instructions: List[Instruction],
    idx: int,
    expr_map: dict,
    tmp_var_names: Set[str],
    tmp_tensor_names: Set[str],
    tmp_gmtensor_names: Set[str],
    helper: CodeHelper,
) -> int:
    inst = instructions[idx]
    if inst.opname != "create_var":
        return -1
    val = inst.kwargs.get("val", None)
    if not isinstance(val, Var) or is_tmp_var(val):
        return -1
    next_idx = _next_emit_index(
        instructions,
        idx + 1,
        tmp_var_names,
        tmp_tensor_names,
        tmp_gmtensor_names,
    )
    if next_idx >= len(instructions):
        return -1
    next_inst = instructions[next_idx]
    if next_inst.opname != "start_loop":
        return -1
    loop_var = next_inst.kwargs.get("var", None)
    if not isinstance(loop_var, Var) or loop_var.name != val.name:
        return -1
    dtype = dtype_to_cpp(val.dtype)
    init_expr = value_to_cpp(val.value, expr_map)
    start = value_to_cpp(next_inst.kwargs.get("start", None), expr_map)
    stop = value_to_cpp(next_inst.kwargs.get("stop", None), expr_map)
    step = value_to_cpp(next_inst.kwargs.get("step", None), expr_map)
    helper(f"for ({dtype} {val.name} = {start}; {val.name} < {stop}; {val.name} += {step}) {{")
    helper.ir()
    return next_idx + 1


def _try_fold_decl_assign(
    instructions: List[Instruction],
    idx: int,
    expr_map: dict,
    tmp_var_names: Set[str],
    tmp_tensor_names: Set[str],
    tmp_gmtensor_names: Set[str],
    helper: CodeHelper,
) -> int:
    inst = instructions[idx]
    if inst.opname != "create_var":
        return -1
    val = inst.kwargs.get("val", None)
    if not isinstance(val, Var) or is_tmp_var(val):
        return -1
    next_idx = _next_emit_index(
        instructions,
        idx + 1,
        tmp_var_names,
        tmp_tensor_names,
        tmp_gmtensor_names,
    )
    if next_idx >= len(instructions):
        return -1
    next_inst = instructions[next_idx]
    if not is_assignment_op(next_inst.opname):
        return -1
    out = next_inst.kwargs.get("out", None)
    if not isinstance(out, Var) or out.name != val.name:
        return -1
    if uses_var_in_operands(next_inst, val.name):
        return -1
    dtype = dtype_to_cpp(val.dtype)
    expr = assignment_expr(next_inst, expr_map)
    helper(f"{dtype} {val.name} = {expr};")
    return next_idx + 1


def translate(instructions: Iterable[Instruction]) -> str:
    instructions = list(instructions)
    expr_map, tmp_var_names, tmp_tensor_names, tmp_gmtensor_names = build_expr_state(instructions)
    helper = CodeHelper()
    handlers = build_handlers()
    unhandled = []
    seen = set()
    idx = 0
    while idx < len(instructions):
        inst = instructions[idx]
        if should_skip_inst(inst, tmp_var_names, tmp_tensor_names, tmp_gmtensor_names):
            idx += 1
            continue
        folded = _try_fold_loop(
            instructions,
            idx,
            expr_map,
            tmp_var_names,
            tmp_tensor_names,
            tmp_gmtensor_names,
            helper,
        )
        if folded != -1:
            idx = folded
            continue
        folded = _try_fold_decl_assign(
            instructions,
            idx,
            expr_map,
            tmp_var_names,
            tmp_tensor_names,
            tmp_gmtensor_names,
            helper,
        )
        if folded != -1:
            idx = folded
            continue
        handler = handlers.get(inst.opname)
        if handler is not None:
            handler(inst, helper, expr_map)
        else:
            opname = inst.opname
            if opname not in seen:
                seen.add(opname)
                unhandled.append(opname)
        idx += 1
    if unhandled:
        helper(f"// Untranslated instructions: {', '.join(unhandled)}")
    return str(helper)


def translate_split(instructions: Iterable[Instruction]) -> Tuple[str, str]:
    cube_insts, vec_insts = split_instructions(instructions)
    print('inserting auto sync instructions...')
    cube_insts = insert_auto_sync(cube_insts, mode='cube')
    print('auto sync instructions inserted cuebe side.')
    vec_insts = insert_auto_sync(vec_insts, mode='vec')
    print('auto sync instructions inserted vec side.')
    return translate(cube_insts), translate(vec_insts)
