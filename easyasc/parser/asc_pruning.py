import re
from typing import Callable, Iterable, List, Optional, Set, Tuple

from .asc_utils import build_expr_state, is_assignment_op, should_skip_inst
from ..utils.instruction import Instruction
from ..utils.var import Expr, Var
from ..utils.Tensor import DBuff, GMTensor, Tensor
from ..utils.events import DEvent, SEvent

# A classifier returns "cube", "vec", or "both" for a single instruction.
ClassifyFn = Callable[[Instruction], str]

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_LOOP_STARTS = ("start_loop", "start_micro_loop")


# ---------------------------
# Block parsing utilities
# ---------------------------

def _same_var(left: object, right: object) -> bool:
    """Return True when two Var-like objects refer to the same logical variable."""
    if left is right:
        return True
    left_name = getattr(left, "name", None)
    right_name = getattr(right, "name", None)
    left_idx = getattr(left, "idx", None)
    right_idx = getattr(right, "idx", None)
    return left_name == right_name and left_idx == right_idx and left_name is not None


def _parse_loop_node(
    instructions: List[Instruction],
    idx: int,
    with_prelude: bool,
) -> Tuple[dict, int]:
    """Parse a loop node, optionally including a leading create_var prelude."""
    prelude = None
    if with_prelude:
        prelude = instructions[idx]
        idx += 1
    start = instructions[idx]
    idx += 1
    body, idx = _parse_block(instructions, idx, {"end_loop"})
    if idx >= len(instructions) or instructions[idx].opname != "end_loop":
        raise ValueError("start_loop/start_micro_loopmissingend_loop")
    end = instructions[idx]
    idx += 1
    return {
        "type": "loop",
        "prelude": prelude,
        "start": start,
        "end": end,
        "body": body,
    }, idx


def _parse_if_chain(
    instructions: List[Instruction],
    idx: int,
) -> Tuple[dict, int]:
    """Parse an if/elif/else chain into a single node with branches."""
    branches = []
    while idx < len(instructions):
        start = instructions[idx]
        if start.opname not in ("start_if", "start_elif", "start_else"):
            break
        idx += 1
        body, idx = _parse_block(instructions, idx, {"end_if"})
        if idx >= len(instructions) or instructions[idx].opname != "end_if":
            raise ValueError("start_if/start_elif/start_elsemissingend_if")
        end = instructions[idx]
        idx += 1
        branches.append({"start": start, "end": end, "body": body})
        if idx >= len(instructions) or instructions[idx].opname not in ("start_elif", "start_else"):
            break
    return {"type": "if", "branches": branches}, idx


def _parse_block(
    instructions: List[Instruction],
    idx: int,
    stop_ops: Optional[Set[str]],
) -> Tuple[List[dict], int]:
    """Parse a flat instruction list into a structured tree of blocks."""
    nodes: List[dict] = []
    while idx < len(instructions) and (stop_ops is None or instructions[idx].opname not in stop_ops):
        inst = instructions[idx]
        # A loop may be preceded by a loop var declaration; keep that with the loop.
        if inst.opname == "create_var" and idx + 1 < len(instructions) and instructions[idx + 1].opname in _LOOP_STARTS:
            loop_var = instructions[idx + 1].kwargs.get("var", None)
            val = inst.kwargs.get("val", None)
            if _same_var(loop_var, val):
                node, idx = _parse_loop_node(instructions, idx, True)
                nodes.append(node)
                continue
        if inst.opname in _LOOP_STARTS:
            node, idx = _parse_loop_node(instructions, idx, False)
            nodes.append(node)
            continue
        if inst.opname == "start_if":
            node, idx = _parse_if_chain(instructions, idx)
            nodes.append(node)
            continue
        nodes.append({"type": "inst", "inst": inst})
        idx += 1
    return nodes, idx


def _emit_instructions(nodes: List[dict]) -> List[Instruction]:
    """Flatten structured nodes back to a linear instruction list."""
    result: List[Instruction] = []
    for node in nodes:
        ntype = node["type"]
        if ntype == "inst":
            result.append(node["inst"])
        elif ntype == "loop":
            prelude = node.get("prelude")
            if prelude is not None:
                result.append(prelude)
            result.append(node["start"])
            result.extend(_emit_instructions(node["body"]))
            result.append(node["end"])
        elif ntype == "if":
            for branch in node["branches"]:
                result.append(branch["start"])
                result.extend(_emit_instructions(branch["body"]))
                result.append(branch["end"])
    return result


# ---------------------------
# 1) Empty block pruning
# ---------------------------

def _is_emitting_inst(
    inst: Instruction,
    tmp_var_names: Set[str],
    tmp_tensor_names: Set[str],
    tmp_gmtensor_names: Set[str],
) -> bool:
    """Check whether an instruction would produce output code."""
    if inst.opname in ("start_loop", "start_micro_loop", "end_loop", "start_if", "start_elif", "start_else", "end_if"):
        return False
    return not should_skip_inst(inst, tmp_var_names, tmp_tensor_names, tmp_gmtensor_names)


def _prune_nodes(
    nodes: List[dict],
    tmp_var_names: Set[str],
    tmp_tensor_names: Set[str],
    tmp_gmtensor_names: Set[str],
) -> Tuple[List[dict], bool]:
    """Remove empty loops/if-chains while preserving blocks that emit code."""
    pruned: List[dict] = []
    has_content = False
    for node in nodes:
        ntype = node["type"]
        if ntype == "inst":
            inst = node["inst"]
            if _is_emitting_inst(inst, tmp_var_names, tmp_tensor_names, tmp_gmtensor_names):
                has_content = True
            pruned.append(node)
        elif ntype == "loop":
            body, body_has_content = _prune_nodes(node["body"], tmp_var_names, tmp_tensor_names, tmp_gmtensor_names)
            if not body_has_content:
                continue
            node["body"] = body
            pruned.append(node)
            has_content = True
        elif ntype == "if":
            any_branch_content = False
            new_branches = []
            for branch in node["branches"]:
                body, body_has_content = _prune_nodes(branch["body"], tmp_var_names, tmp_tensor_names, tmp_gmtensor_names)
                branch["body"] = body
                new_branches.append(branch)
                if body_has_content:
                    any_branch_content = True
            if not any_branch_content:
                continue
            node["branches"] = new_branches
            pruned.append(node)
            has_content = True
    return pruned, has_content


def prune_empty_blocks(instructions: List[Instruction]) -> List[Instruction]:
    """
    Remove loops/if-chains that do not emit any code.
    This pass only analyzes structural blocks and does not change semantics.
    """
    _, tmp_var_names, tmp_tensor_names, tmp_gmtensor_names = build_expr_state(instructions)
    nodes, idx = _parse_block(list(instructions), 0, None)
    if idx != len(instructions):
        raise ValueError("instruction block parsing did not fully consume instructions")
    pruned_nodes, _ = _prune_nodes(nodes, tmp_var_names, tmp_tensor_names, tmp_gmtensor_names)
    return _emit_instructions(pruned_nodes)


# ---------------------------
# 2) Unused declaration pruning
# ---------------------------

def _mark_used_from_value(
    value: object,
    used_ids: Set[int],
    tmp_tensor_names: Set[str],
    used_tmp_tensors: Set[str],
    tmp_gmtensor_names: Set[str],
) -> None:
    """Mark objects referenced by instruction operands as used."""
    if isinstance(value, Tensor):
        if value.name in tmp_tensor_names:
            used_tmp_tensors.add(value.name)
            return
        used_ids.add(id(value))
        return
    if isinstance(value, GMTensor):
        if value.name in tmp_gmtensor_names:
            return
        used_ids.add(id(value))
        return
    if isinstance(value, (DBuff, SEvent, DEvent)):
        used_ids.add(id(value))
        return
    if isinstance(value, dict):
        for item in value.values():
            _mark_used_from_value(item, used_ids, tmp_tensor_names, used_tmp_tensors, tmp_gmtensor_names)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _mark_used_from_value(item, used_ids, tmp_tensor_names, used_tmp_tensors, tmp_gmtensor_names)
        return


def _build_tmp_tensor_sources(
    instructions: List[Instruction],
    tmp_tensor_names: Set[str],
) -> dict[str, object]:
    """Map temporary tensor names to their source DBuff/Tensor, if any."""
    sources: dict[str, object] = {}
    for inst in instructions:
        if inst.opname == "get_buf":
            out = inst.kwargs.get("out", None)
            if isinstance(out, Tensor) and out.name in tmp_tensor_names:
                sources[out.name] = inst.kwargs.get("buf", None)
        elif inst.opname in ("slice_tensor", "micro_slice_tensor"):
            out = inst.kwargs.get("out", None)
            if isinstance(out, Tensor) and out.name in tmp_tensor_names:
                sources[out.name] = inst.kwargs.get("src", None)
    return sources


def _collect_used_ids(
    instructions: List[Instruction],
    tmp_var_names: Set[str],
    tmp_tensor_names: Set[str],
    tmp_gmtensor_names: Set[str],
) -> Set[int]:
    """
    Collect object ids (DBuff/Tensor/GMTensor/SEvent/DEvent) that are referenced
    by any emitting instruction. This allows us to drop unused create_* decls.
    """
    used_ids: Set[int] = set()
    used_tmp_tensors: Set[str] = set()
    for inst in instructions:
        # Skip declarations; they are the ones we're deciding to keep or remove.
        if inst.opname in ("create_dbuf", "create_tensor", "create_gm_tensor", "create_sevent", "create_devent"):
            continue
        # Skip tmp-only helper instructions, but preserve their sources.
        if should_skip_inst(inst, tmp_var_names, tmp_tensor_names, tmp_gmtensor_names):
            if inst.opname == "slice_gm_tensor":
                src = inst.kwargs.get("src", None)
                if isinstance(src, GMTensor) and src.name not in tmp_gmtensor_names:
                    used_ids.add(id(src))
            continue
        for value in inst.kwargs.values():
            _mark_used_from_value(value, used_ids, tmp_tensor_names, used_tmp_tensors, tmp_gmtensor_names)

    # Propagate usage from temporary tensors to their original buffers.
    sources = _build_tmp_tensor_sources(instructions, tmp_tensor_names)
    pending = list(used_tmp_tensors)
    seen_tmp = set(pending)
    while pending:
        name = pending.pop()
        src = sources.get(name, None)
        if src is None:
            continue
        if isinstance(src, Tensor) and src.name in tmp_tensor_names:
            if src.name not in seen_tmp:
                seen_tmp.add(src.name)
                pending.append(src.name)
            continue
        if isinstance(src, (DBuff, Tensor, GMTensor, SEvent, DEvent)):
            used_ids.add(id(src))
    return used_ids


def prune_unused_decls(instructions: List[Instruction]) -> List[Instruction]:
    """
    Remove create_* declarations for objects that are never referenced
    by any emitting instruction in this instruction list.
    """
    _, tmp_var_names, tmp_tensor_names, tmp_gmtensor_names = build_expr_state(instructions)
    _ = tmp_var_names
    used_ids = _collect_used_ids(instructions, tmp_var_names, tmp_tensor_names, tmp_gmtensor_names)
    pruned: List[Instruction] = []
    for inst in instructions:
        if inst.opname in ("create_gm_tensor", "create_sevent", "create_devent"):
            val = inst.kwargs.get("val", None)
            if val is None:
                pruned.append(inst)
                continue
            # Temporary tensors/gmtensors are inlined and should not emit declarations.
            if isinstance(val, Tensor) and val.name in tmp_tensor_names:
                continue
            if isinstance(val, GMTensor) and val.name in tmp_gmtensor_names:
                continue
            if id(val) not in used_ids:
                continue
        pruned.append(inst)
    return pruned


# ---------------------------
# 3) Unused Var pruning
# ---------------------------

def _collect_known_var_names(instructions: List[Instruction]) -> Set[str]:
    """Collect all Var names declared in this instruction list."""
    names: Set[str] = set()
    for inst in instructions:
        if inst.opname == "create_var":
            val = inst.kwargs.get("val", None)
            if isinstance(val, Var):
                names.add(val.name)
    return names


def _extract_var_names_from_value(value: object, known_names: Set[str], out: Set[str]) -> None:
    """Extract Var names from a value (Var / Expr / containers)."""
    if isinstance(value, Var):
        out.add(value.name)
        return
    if isinstance(value, Expr):
        for name in _IDENT_RE.findall(value.expr):
            if name in known_names:
                out.add(name)
        return
    if isinstance(value, (Tensor, DBuff, GMTensor, SEvent, DEvent)):
        return
    if isinstance(value, dict):
        for item in value.values():
            _extract_var_names_from_value(item, known_names, out)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _extract_var_names_from_value(item, known_names, out)
        return


def _inst_is_side_specific(inst: Instruction, side: str, classify_inst: ClassifyFn) -> bool:
    """Return True if this instruction is specific to the given side."""
    side_tag = classify_inst(inst)
    return side_tag == side or side_tag == "both"


def _collect_var_deps(
    instructions: List[Instruction],
    known_names: Set[str],
) -> dict[str, Set[str]]:
    """Build a dependency map: out_var -> input_vars."""
    deps: dict[str, Set[str]] = {}
    for inst in instructions:
        if not is_assignment_op(inst.opname):
            continue
        out = inst.kwargs.get("out", None)
        if not isinstance(out, Var):
            continue
        inputs: Set[str] = set()
        if inst.opname in ("CeilDiv", "Min", "Max", "var_mul", "var_div", "var_add", "var_sub"):
            _extract_var_names_from_value(inst.kwargs.get("a", None), known_names, inputs)
            _extract_var_names_from_value(inst.kwargs.get("b", None), known_names, inputs)
        elif inst.opname in ("scalar_sqrt", "Align16", "Align32", "Align64", "Align128", "Align256"):
            _extract_var_names_from_value(inst.kwargs.get("a", None), known_names, inputs)
        deps.setdefault(out.name, set()).update(inputs)
    return deps


def _collect_seed_usage(
    nodes: List[dict],
    side: str,
    classify_inst: ClassifyFn,
    known_names: Set[str],
) -> Tuple[Set[str], Set[int], Set[int]]:
    """
    Collect seed usage from blocks that actually emit side-specific code.
    We seed:
      - Vars used directly by side-specific instructions or by the loop/if that
        encloses them.
      - Tensors/GMTensors passed directly to side-specific instructions.
    """
    seed_vars: Set[str] = set()
    seed_tensors: Set[int] = set()
    seed_gmtensors: Set[int] = set()

    def _collect_from_value(value: object) -> None:
        if isinstance(value, Tensor):
            seed_tensors.add(id(value))
            return
        if isinstance(value, GMTensor):
            seed_gmtensors.add(id(value))
            return
        _extract_var_names_from_value(value, known_names, seed_vars)

    def _walk(block: List[dict]) -> Tuple[Set[str], bool]:
        block_used: Set[str] = set()
        has_side = False
        for node in block:
            ntype = node["type"]
            if ntype == "inst":
                inst = node["inst"]
                if _inst_is_side_specific(inst, side, classify_inst):
                    if inst.opname == "create_var":
                        continue
                    has_side = True
                    for value in inst.kwargs.values():
                        _collect_from_value(value)
                        _extract_var_names_from_value(value, known_names, block_used)
            elif ntype == "loop":
                body_used, body_has = _walk(node["body"])
                if body_has:
                    has_side = True
                    block_used.update(body_used)
                    start_inst = node["start"]
                    for value in start_inst.kwargs.values():
                        _extract_var_names_from_value(value, known_names, block_used)
            elif ntype == "if":
                branch_used: Set[str] = set()
                any_branch = False
                for branch in node["branches"]:
                    used, has = _walk(branch["body"])
                    branch_used.update(used)
                    if has:
                        any_branch = True
                if any_branch:
                    has_side = True
                    block_used.update(branch_used)
                    for branch in node["branches"]:
                        start_inst = branch["start"]
                        for value in start_inst.kwargs.values():
                            _extract_var_names_from_value(value, known_names, block_used)
        return block_used, has_side

    vars_in_blocks, _ = _walk(nodes)
    seed_vars.update(vars_in_blocks)
    return seed_vars, seed_tensors, seed_gmtensors


def _build_tensor_defs(instructions: List[Instruction]) -> dict[int, Instruction]:
    """Map Tensor id -> defining instruction (get_buf / slice_tensor / micro_slice_tensor / reinterpret)."""
    defs: dict[int, Instruction] = {}
    for inst in instructions:
        if inst.opname in ("get_buf", "slice_tensor", "micro_slice_tensor"):
            out = inst.kwargs.get("out", None)
            if isinstance(out, Tensor):
                defs[id(out)] = inst
        elif inst.opname == "reinterpret":
            out = inst.kwargs.get("dst", None)
            if isinstance(out, Tensor):
                defs[id(out)] = inst
    return defs


def _build_gmtensor_defs(instructions: List[Instruction]) -> dict[int, Instruction]:
    """Map GMTensor id -> defining instruction (slice_gm_tensor)."""
    defs: dict[int, Instruction] = {}
    for inst in instructions:
        if inst.opname == "slice_gm_tensor":
            out = inst.kwargs.get("out", None)
            if isinstance(out, GMTensor):
                defs[id(out)] = inst
    return defs


def prune_unused_vars(
    instructions: List[Instruction],
    side: str,
    classify_inst: ClassifyFn,
) -> List[Instruction]:
    """
    Remove Vars that do not affect any side-specific instruction.

    Strategy:
      1) Identify seed Vars/Tensors/GMTensors required by side-specific ops.
      2) Propagate Var usage through tensor/gmtensor definitions.
      3) Expand Var usage via Var->Var dependency graph.
      4) Drop unused Var declarations and their assignment ops.
      5) Drop unused temp tensor/gmtensor defs (get_buf/slice_*).
    """
    known_names = _collect_known_var_names(instructions)
    nodes, idx = _parse_block(list(instructions), 0, None)
    if idx != len(instructions):
        raise ValueError("instruction block parsing did not fully consume instructions")
    seed_vars, seed_tensors, seed_gmtensors = _collect_seed_usage(nodes, side, classify_inst, known_names)

    tensor_defs = _build_tensor_defs(instructions)
    gmtensor_defs = _build_gmtensor_defs(instructions)
    used_tensors = set(seed_tensors)
    used_gmtensors = set(seed_gmtensors)

    # Walk tensor defs and pull in Vars used by indexing/offsets.
    pending_tensors = list(used_tensors)
    seen_tensors = set(pending_tensors)
    while pending_tensors:
        tid = pending_tensors.pop()
        inst = tensor_defs.get(tid)
        if inst is None:
            continue
        if inst.opname == "get_buf":
            _extract_var_names_from_value(inst.kwargs.get("index", None), known_names, seed_vars)
        elif inst.opname in ("slice_tensor", "micro_slice_tensor"):
            _extract_var_names_from_value(inst.kwargs.get("offset", None), known_names, seed_vars)
            _extract_var_names_from_value(inst.kwargs.get("span", None), known_names, seed_vars)
            _extract_var_names_from_value(inst.kwargs.get("step", None), known_names, seed_vars)
            src = inst.kwargs.get("src", None)
            if isinstance(src, Tensor):
                sid = id(src)
                if sid not in seen_tensors:
                    seen_tensors.add(sid)
                    pending_tensors.append(sid)
                    used_tensors.add(sid)
        elif inst.opname == "reinterpret":
            # Reinterpret keeps data view; usage must propagate to the source tensor.
            src = inst.kwargs.get("src", None)
            if isinstance(src, Tensor):
                sid = id(src)
                if sid not in seen_tensors:
                    seen_tensors.add(sid)
                    pending_tensors.append(sid)
                    used_tensors.add(sid)

    # Walk gmtensor defs and pull in Vars used by offsets/shapes.
    pending_gmtensors = list(used_gmtensors)
    seen_gmtensors = set(pending_gmtensors)
    while pending_gmtensors:
        gid = pending_gmtensors.pop()
        inst = gmtensor_defs.get(gid)
        if inst is None:
            continue
        _extract_var_names_from_value(inst.kwargs.get("offset", None), known_names, seed_vars)
        _extract_var_names_from_value(inst.kwargs.get("shape", None), known_names, seed_vars)
        src = inst.kwargs.get("src", None)
        if isinstance(src, GMTensor):
            sid = id(src)
            if sid not in seen_gmtensors:
                seen_gmtensors.add(sid)
                pending_gmtensors.append(sid)
                used_gmtensors.add(sid)

    # Expand var usage through assignment dependencies.
    deps = _collect_var_deps(instructions, known_names)
    used_vars = set(seed_vars)
    changed = True
    while changed:
        changed = False
        for out_name, inputs in deps.items():
            if out_name in used_vars:
                for name in inputs:
                    if name not in used_vars:
                        used_vars.add(name)
                        changed = True

    _, tmp_var_names, tmp_tensor_names, tmp_gmtensor_names = build_expr_state(instructions)
    _ = tmp_var_names
    _ = tmp_tensor_names
    _ = tmp_gmtensor_names
    declared_vars = {name for name in known_names if name not in tmp_var_names}
    pruned_vars = declared_vars - used_vars

    pruned: List[Instruction] = []
    for inst in instructions:
        if inst.opname == "create_var":
            val = inst.kwargs.get("val", None)
            if isinstance(val, Var) and val.name in pruned_vars:
                continue
        if is_assignment_op(inst.opname):
            out = inst.kwargs.get("out", None)
            if isinstance(out, Var) and out.name in pruned_vars:
                continue
        if inst.opname in ("get_buf", "slice_tensor", "micro_slice_tensor"):
            out = inst.kwargs.get("out", None)
            if isinstance(out, Tensor) and id(out) not in used_tensors:
                continue
        if inst.opname == "reinterpret":
            out = inst.kwargs.get("dst", None)
            if isinstance(out, Tensor) and id(out) not in used_tensors:
                continue
        if inst.opname == "slice_gm_tensor":
            out = inst.kwargs.get("out", None)
            if isinstance(out, GMTensor) and id(out) not in used_gmtensors:
                continue
        pruned.append(inst)
    return pruned
