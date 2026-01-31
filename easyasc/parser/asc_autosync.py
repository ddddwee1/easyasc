from typing import Dict, List, Optional, Set, Tuple
import os
import re
from functools import lru_cache

from .asc_utils import build_expr_state, should_skip_inst
from ..utils.events import DEvent, SEvent
from .asc_handlers import build_handlers
from ..utils.Tensor import Tensor
from ..utils.instruction import Instruction
from ..utils.pipe import Pipe


_AUTO_SYNC_START = "start_auto_sync"
_AUTO_SYNC_END = "end_auto_sync"
_MTE2_OPS = {"gm_to_l1_nd2nz", "GM2UBPAD", "gm_to_ub_pad"}
_MTE1_OPS = {"l1_to_l0"}
_M_OPS = {"mmad"}
_FIX_OPS = {"l0c_to_gm_nz2nd"}
_MTE3_OPS = {"UB2GMPAD", "ub_to_gm_pad"}
_EVENT_OPS = {
    "create_sevent",
    "create_devent",
    "event_set",
    "event_wait",
    "event_setall",
    "event_release",
}

_VEC_INSTRUCTION_RE = re.compile(r"Instruction\(\s*['\"]([A-Za-z0-9_]+)['\"]")


def _collect_opnames_from_file(path: str) -> Set[str]:
    try:
        with open(path, "r") as f:
            text = f.read()
    except OSError:
        return set()
    return set(_VEC_INSTRUCTION_RE.findall(text))


def _collect_opnames_from_dir(path: str) -> Set[str]:
    names: Set[str] = set()
    for root, _, files in os.walk(path):
        for filename in files:
            if filename.endswith(".py"):
                names |= _collect_opnames_from_file(os.path.join(root, filename))
    return names


@lru_cache(maxsize=1)
def _get_v_ops() -> Set[str]:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    vec_dir = os.path.join(base_dir, "stub_functions", "vec")
    ops = _collect_opnames_from_dir(vec_dir)
    handler_map = build_handlers()
    for opname, handler in handler_map.items():
        module = getattr(handler, "__module__", "")
        if ".vec_" in module:
            ops.add(opname)
    ops.discard("GM2UBPAD")
    ops.discard("UB2GMPAD")
    return ops


def _compute_in_segment(instructions: List[Instruction]) -> List[bool]:
    has_markers = any(inst.opname in (_AUTO_SYNC_START, _AUTO_SYNC_END) for inst in instructions)
    if not has_markers:
        return [True] * len(instructions)
    in_segment: List[bool] = []
    depth = 0
    for inst in instructions:
        if inst.opname == _AUTO_SYNC_START:
            depth += 1
            in_segment.append(False)
            continue
        if inst.opname == _AUTO_SYNC_END:
            in_segment.append(False)
            if depth > 0:
                depth -= 1
            continue
        in_segment.append(depth > 0)
    return in_segment


def _collect_event_names(instructions: List[Instruction]) -> Set[str]:
    names: Set[str] = set()
    for inst in instructions:
        for val in inst.kwargs.values():
            if isinstance(val, (DEvent, SEvent)):
                names.add(val.name)
    return names


def _sevent_needed_for_consumer(inst: Instruction) -> bool:
    for tensor in inst.kwargs.values():
        if not isinstance(tensor, Tensor):
            continue
        source_buf = getattr(tensor, "source_buf", None)
        if source_buf is None or isinstance(source_buf, Tensor):
            return True
    return False


def _sevent_needed_for_consumer_src_only(inst: Instruction) -> bool:
    src = inst.kwargs.get("src", None)
    if isinstance(src, Tensor):
        source_buf = getattr(src, "source_buf", None)
        return source_buf is None or isinstance(source_buf, Tensor)
    return False


def _sevent_needed_for_gm_to_l1(inst: Instruction) -> bool:
    if inst.opname != "gm_to_l1_nd2nz":
        return False
    dst = inst.kwargs.get("dst", None)
    if isinstance(dst, Tensor):
        source_buf = getattr(dst, "source_buf", None)
        return source_buf is None or isinstance(source_buf, Tensor)
    return False


def _compute_segments(in_segment: List[bool]) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    idx = 0
    n = len(in_segment)
    while idx < n:
        if not in_segment[idx]:
            idx += 1
            continue
        start = idx
        while idx + 1 < n and in_segment[idx + 1]:
            idx += 1
        segments.append((start, idx))
        idx += 1
    return segments


def _collect_if_chains(instructions: List[Instruction]) -> List[Tuple[int, int]]:
    chains: List[Tuple[int, int]] = []
    idx = 0
    n = len(instructions)
    while idx < n:
        if instructions[idx].opname != "start_if":
            idx += 1
            continue
        chain_start = idx
        idx = _scan_to_end_if(instructions, idx)
        chain_end = idx - 1
        while idx < n and instructions[idx].opname in ("start_elif", "start_else"):
            idx = _scan_to_end_if(instructions, idx)
            chain_end = idx - 1
        chains.append((chain_start, chain_end))
    return chains


def _scan_to_end_if(instructions: List[Instruction], start_idx: int) -> int:
    depth = 1
    idx = start_idx + 1
    n = len(instructions)
    while idx < n:
        opname = instructions[idx].opname
        if opname == "start_if":
            depth += 1
        elif opname == "end_if":
            depth -= 1
            if depth == 0:
                return idx + 1
        idx += 1
    return n


def _build_chain_map(chains: List[Tuple[int, int]], n: int) -> List[Optional[Tuple[int, int]]]:
    chain_for_index: List[Optional[Tuple[int, int]]] = [None] * n
    chains_sorted = sorted(chains, key=lambda item: (item[1] - item[0]))
    for start, end in chains_sorted:
        for idx in range(start, end + 1):
            if chain_for_index[idx] is None:
                chain_for_index[idx] = (start, end)
    return chain_for_index


def _alloc_event_pair(
    existing: Set[str],
    prefix: str,
    start_idx: int,
    level: str,
) -> tuple[int, str, str]:
    idx = start_idx
    while True:
        valid_name = f"{prefix}{idx}_{level}_valid"
        ready_name = f"{prefix}{idx}_{level}_ready"
        if valid_name not in existing and ready_name not in existing:
            existing.add(valid_name)
            existing.add(ready_name)
            return idx, valid_name, ready_name
        idx += 1


def _insert_auto_sync_for_pipe(
    insts: List[Instruction],
    producer_ops: Set[str],
    consumer_ops: Set[str],
    producer_pipe: Pipe,
    consumer_pipe: Pipe,
    name_prefix: str,
    name_level: str,
    consumer_sevent_predicate=_sevent_needed_for_consumer,
) -> List[Instruction]:
    if not insts:
        return insts
    _, tmp_var_names, tmp_tensor_names, tmp_gmtensor_names = build_expr_state(insts)
    in_segment = _compute_in_segment(insts)
    segments = _compute_segments(in_segment)
    active_seg_ids: Set[int] = set()
    for seg_id, (start, end) in enumerate(segments):
        has_producer = False
        has_consumer = False
        for idx in range(start, end + 1):
            if insts[idx].opname in producer_ops:
                has_producer = True
            if insts[idx].opname in consumer_ops:
                has_consumer = True
            if has_producer and has_consumer:
                break
        if has_producer and has_consumer:
            active_seg_ids.add(seg_id)
    if not active_seg_ids:
        return insts

    existing_names = _collect_event_names(insts)
    segments = segments
    seg_id_per_index = [-1] * len(insts)
    depth_per_index = [-1] * len(insts)
    segment_depths: Dict[int, List[int]] = {}
    max_pairs = 0
    for seg_id, (start, end) in enumerate(segments):
        if seg_id not in active_seg_ids:
            continue
        depth = 0
        depths_with_producer: Set[int] = set()
        depths_with_consumer: Set[int] = set()
        for idx in range(start, end + 1):
            seg_id_per_index[idx] = seg_id
            if insts[idx].opname == "start_loop":
                depth += 1
            depth_per_index[idx] = depth
            if insts[idx].opname in producer_ops:
                depths_with_producer.add(depth)
            if insts[idx].opname in consumer_ops:
                depths_with_consumer.add(depth)
            if insts[idx].opname == "end_loop" and depth > 0:
                depth -= 1
        if name_level == "ubout":
            sorted_depths = sorted(depths_with_consumer) or sorted(depths_with_producer)
        else:
            sorted_depths = sorted(depths_with_producer)
        segment_depths[seg_id] = sorted_depths
        if len(sorted_depths) > max_pairs:
            max_pairs = len(sorted_depths)

    if max_pairs == 0:
        return insts

    segment_depth_to_rank: Dict[int, Dict[int, int]] = {}
    for seg_id, depths in segment_depths.items():
        segment_depth_to_rank[seg_id] = {depth: rank for rank, depth in enumerate(depths)}

    def _is_ignorable(inst: Instruction) -> bool:
        if inst.opname in _EVENT_OPS:
            return True
        return should_skip_inst(inst, tmp_var_names, tmp_tensor_names, tmp_gmtensor_names)

    def _prev_significant_index(cur_idx: int) -> int:
        idx = cur_idx - 1
        while idx >= 0:
            if not in_segment[idx] or seg_id_per_index[idx] == -1:
                break
            if _is_ignorable(insts[idx]):
                idx -= 1
                continue
            return idx
        return -1

    def _next_significant_index(cur_idx: int) -> int:
        idx = cur_idx + 1
        while idx < len(insts):
            if not in_segment[idx] or seg_id_per_index[idx] == -1:
                break
            if _is_ignorable(insts[idx]):
                idx += 1
                continue
            return idx
        return -1

    chains = _collect_if_chains(insts)
    chain_for_index = _build_chain_map(chains, len(insts))

    pair_id_per_index = [-1] * len(insts)
    for idx, inst in enumerate(insts):
        if not in_segment[idx] or seg_id_per_index[idx] == -1:
            continue
        seg_id = seg_id_per_index[idx]
        depths = segment_depths.get(seg_id, [])
        if not depths:
            pair_id_per_index[idx] = 0
            continue
        current_depth = depth_per_index[idx]
        chosen_depth = None
        for depth in depths:
            if depth <= current_depth:
                chosen_depth = depth
            else:
                break
        if chosen_depth is None:
            chosen_depth = depths[0]
        pair_id_per_index[idx] = segment_depth_to_rank[seg_id][chosen_depth]

    use_sevent_for_seg_pair: Dict[Tuple[int, int], bool] = {}
    for idx, inst in enumerate(insts):
        if not in_segment[idx] or seg_id_per_index[idx] == -1:
            continue
        seg_id = seg_id_per_index[idx]
        pair_id = pair_id_per_index[idx]
        if inst.opname in consumer_ops and consumer_sevent_predicate(inst):
            use_sevent_for_seg_pair[(seg_id, pair_id)] = True
        if name_level == "l1" and _sevent_needed_for_gm_to_l1(inst):
            use_sevent_for_seg_pair[(seg_id, pair_id)] = True

    de_needed = [False] * max_pairs
    se_needed = [False] * max_pairs
    for seg_id, depths in segment_depths.items():
        if seg_id not in active_seg_ids:
            continue
        for pair_id in range(len(depths)):
            if use_sevent_for_seg_pair.get((seg_id, pair_id), False):
                se_needed[pair_id] = True
            else:
                de_needed[pair_id] = True

    next_idx_for_prefix: Dict[str, int] = {
        name_prefix: 1,
        "_tmp_sevent": 1,
    }
    de_pool: List[Optional[Tuple[DEvent, DEvent]]] = [None] * max_pairs
    se_pool: List[Optional[Tuple[SEvent, SEvent]]] = [None] * max_pairs
    for pair_id in range(max_pairs):
        if de_needed[pair_id]:
            start_idx = next_idx_for_prefix.get(name_prefix, 1)
            pair_idx, valid_name, ready_name = _alloc_event_pair(
                existing_names,
                name_prefix,
                start_idx,
                name_level,
            )
            next_idx_for_prefix[name_prefix] = pair_idx + 1
            de_pool[pair_id] = (
                DEvent(consumer_pipe, producer_pipe, name=valid_name),
                DEvent(producer_pipe, consumer_pipe, name=ready_name),
            )
        if se_needed[pair_id]:
            start_idx = next_idx_for_prefix.get("_tmp_sevent", 1)
            pair_idx, valid_name, ready_name = _alloc_event_pair(
                existing_names,
                "_tmp_sevent",
                start_idx,
                name_level,
            )
            next_idx_for_prefix["_tmp_sevent"] = pair_idx + 1
            se_pool[pair_id] = (
                SEvent(consumer_pipe, producer_pipe, name=valid_name),
                SEvent(producer_pipe, consumer_pipe, name=ready_name),
            )

    def _event_for(seg_id: int, pair_id: int, which: int) -> object:
        if use_sevent_for_seg_pair.get((seg_id, pair_id), False):
            event_pair = se_pool[pair_id]
            if event_pair is None:
                event_pair = de_pool[pair_id]
        else:
            event_pair = de_pool[pair_id]
            if event_pair is None:
                event_pair = se_pool[pair_id]
        if event_pair is None:
            raise RuntimeError("auto_sync event pool missing for pair")
        return event_pair[which]

    new_insts: List[Instruction] = []
    for pair_id in range(max_pairs):
        if de_pool[pair_id] is not None:
            valid_event, ready_event = de_pool[pair_id]
            new_insts.append(Instruction("create_devent", val=valid_event))
            new_insts.append(Instruction("create_devent", val=ready_event))
        if se_pool[pair_id] is not None:
            valid_event, ready_event = se_pool[pair_id]
            new_insts.append(Instruction("create_sevent", val=valid_event))
            new_insts.append(Instruction("create_sevent", val=ready_event))

    loop_end_for_start: Dict[int, int] = {}
    loop_stack: List[int] = []
    for idx, inst in enumerate(insts):
        if not in_segment[idx] or seg_id_per_index[idx] == -1:
            continue
        if inst.opname == "start_loop":
            loop_stack.append(idx)
        elif inst.opname == "end_loop" and loop_stack:
            start_idx = loop_stack.pop()
            loop_end_for_start[start_idx] = idx

    loop_stack_per_index: List[Tuple[int, ...]] = [tuple()] * len(insts)
    loop_stack = []
    for idx, inst in enumerate(insts):
        if in_segment[idx] and seg_id_per_index[idx] != -1 and inst.opname == "start_loop":
            loop_stack.append(idx)
        if in_segment[idx] and seg_id_per_index[idx] != -1:
            loop_stack_per_index[idx] = tuple(loop_stack)
        if in_segment[idx] and seg_id_per_index[idx] != -1 and inst.opname == "end_loop" and loop_stack:
            loop_stack.pop()

    min_consumer_depth: Dict[Tuple[int, int], int] = {}
    last_consumer_idx: Dict[Tuple[int, int], int] = {}
    for idx, inst in enumerate(insts):
        if not in_segment[idx] or inst.opname not in consumer_ops:
            continue
        seg_id = seg_id_per_index[idx]
        pair_id = pair_id_per_index[idx]
        key = (seg_id, pair_id)
        depth = depth_per_index[idx]
        prev = min_consumer_depth.get(key)
        if prev is None or depth < prev:
            min_consumer_depth[key] = depth
        last_consumer_idx[key] = idx

    first_producer_idx: Dict[Tuple[int, int], int] = {}
    last_producer_idx: Dict[Tuple[int, int], int] = {}
    last_shallow_producer_idx: Dict[Tuple[int, int], int] = {}
    min_producer_depth: Dict[Tuple[int, int], int] = {}
    for idx, inst in enumerate(insts):
        if not in_segment[idx] or seg_id_per_index[idx] == -1 or inst.opname not in producer_ops:
            continue
        seg_id = seg_id_per_index[idx]
        pair_id = pair_id_per_index[idx]
        seg_pair = (seg_id, pair_id)
        first_producer_idx.setdefault(seg_pair, idx)
        last_producer_idx[seg_pair] = idx
        prod_depth = depth_per_index[idx]
        prev_min = min_producer_depth.get(seg_pair)
        if prev_min is None or prod_depth < prev_min:
            min_producer_depth[seg_pair] = prod_depth
        depths = segment_depths.get(seg_id, [])
        pair_depth = depths[pair_id] if pair_id < len(depths) else 0
        if prod_depth <= pair_depth:
            last_shallow_producer_idx[seg_pair] = idx

    insert_before: Dict[int, List[Instruction]] = {}
    insert_after: Dict[int, List[Instruction]] = {}
    seen_wait: Set[Tuple[int, int]] = set()
    seen_pre_valid: Set[Tuple[int, int]] = set()
    seen_ready: Set[Tuple[int, int]] = set()
    seen_ready_pairs: Set[Tuple[int, int]] = set()
    block_start_for_pair: Dict[int, int] = {}
    loop_stack = []
    for idx, inst in enumerate(insts):
        if inst.opname == "start_loop" and in_segment[idx] and seg_id_per_index[idx] != -1:
            loop_stack.append(idx)
        elif inst.opname == "end_loop" and in_segment[idx] and seg_id_per_index[idx] != -1 and loop_stack:
            loop_stack.pop()
        if not in_segment[idx] or seg_id_per_index[idx] == -1 or inst.opname not in producer_ops:
            continue
        pair_id = pair_id_per_index[idx]
        seg_id = seg_id_per_index[idx]
        seg_pair = (seg_id, pair_id)
        depths = segment_depths.get(seg_id, [])
        pair_depth = depths[pair_id] if pair_id < len(depths) else 0
        target_depth: Optional[int] = None
        if name_level in ("l0", "fix", "ubout"):
            consumer_depth = min_consumer_depth.get((seg_id, pair_id))
            if consumer_depth is not None and consumer_depth < pair_depth:
                target_depth = consumer_depth
        prev_idx = _prev_significant_index(idx)
        same_pair_prev = (
            prev_idx != -1
            and pair_id_per_index[prev_idx] == pair_id
            and insts[prev_idx].opname in producer_ops
        )
        if not same_pair_prev:
            block_start_for_pair[pair_id] = idx
        next_idx = _next_significant_index(idx)
        same_pair_next = (
            next_idx != -1
            and pair_id_per_index[next_idx] == pair_id
            and insts[next_idx].opname in producer_ops
        )
        if not same_pair_next:
            start_idx = block_start_for_pair.get(pair_id, idx)
            end_idx = idx
            wait_idx = start_idx
            ready_idx = end_idx
            if target_depth is not None:
                if len(loop_stack) > target_depth:
                    outer_start = loop_stack[target_depth]
                    outer_end = loop_end_for_start.get(outer_start)
                    if outer_end is not None:
                        wait_idx = outer_start
                        ready_idx = outer_end
            else:
                chain = chain_for_index[start_idx] or chain_for_index[end_idx]
                if chain is not None:
                    wait_idx, ready_idx = chain
            if name_level == "ubout":
                continue
            wait_key = (pair_id, wait_idx)
            if wait_key not in seen_wait:
                seen_wait.add(wait_key)
                insert_before.setdefault(wait_idx, []).append(
                    Instruction("event_wait", event=_event_for(seg_id, pair_id, 0))
                )
            ready_key = (pair_id, ready_idx)
            if ready_key not in seen_ready:
                seen_ready.add(ready_key)
                seen_ready_pairs.add(seg_pair)
                insert_after.setdefault(ready_idx, []).append(
                    Instruction("event_set", event=_event_for(seg_id, pair_id, 1))
                )
                insert_after.setdefault(ready_idx, []).append(
                    Instruction("event_wait", event=_event_for(seg_id, pair_id, 1))
                )

    if name_level == "fix":
        for seg_pair, last_idx in last_producer_idx.items():
            if seg_pair not in min_consumer_depth:
                continue
            if seg_pair in seen_ready_pairs:
                continue
            seg_id, pair_id = seg_pair
            start_idx = first_producer_idx.get(seg_pair, last_idx)
            ready_idx = last_idx
            chain = chain_for_index[last_idx] or chain_for_index[start_idx]
            if chain is not None and seg_id_per_index[chain[1]] != -1:
                ready_idx = chain[1]
            depths = segment_depths.get(seg_id, [])
            pair_depth = depths[pair_id] if pair_id < len(depths) else 0
            target_depth = None
            consumer_depth = min_consumer_depth.get(seg_pair)
            if consumer_depth is not None and consumer_depth < pair_depth:
                target_depth = consumer_depth
            if target_depth is not None:
                stack = loop_stack_per_index[last_idx]
                if len(stack) > target_depth:
                    outer_start = stack[target_depth]
                    outer_end = loop_end_for_start.get(outer_start)
                    if outer_end is not None:
                        ready_idx = outer_end
            ready_key = (pair_id, ready_idx)
            if ready_key not in seen_ready:
                seen_ready.add(ready_key)
                insert_after.setdefault(ready_idx, []).append(
                    Instruction("event_set", event=_event_for(seg_id, pair_id, 1))
                )
                insert_after.setdefault(ready_idx, []).append(
                    Instruction("event_wait", event=_event_for(seg_id, pair_id, 1))
                )

    if name_level == "ubout":
        for idx, inst in enumerate(insts):
            if not in_segment[idx] or seg_id_per_index[idx] == -1 or inst.opname not in consumer_ops:
                continue
            seg_id = seg_id_per_index[idx]
            pair_id = pair_id_per_index[idx]
            seg_pair = (seg_id, pair_id)
            depths = segment_depths.get(seg_id, [])
            pair_depth = depths[pair_id] if pair_id < len(depths) else 0
            # Insert pre-valid before the first producer in the consumer's scope.
            scope_start = segments[seg_id][0]
            if pair_depth > 0:
                stack = loop_stack_per_index[idx]
                if len(stack) >= pair_depth:
                    scope_start = stack[pair_depth - 1]
            search_start = scope_start + 1
            first_prod_idx = None
            for j in range(search_start, idx):
                if not in_segment[j] or seg_id_per_index[j] != seg_id:
                    continue
                if insts[j].opname in producer_ops:
                    first_prod_idx = j
                    break
            if first_prod_idx is not None:
                pre_valid_idx = first_prod_idx
                if depth_per_index[first_prod_idx] > pair_depth:
                    stack = loop_stack_per_index[first_prod_idx]
                    if len(stack) > pair_depth:
                        pre_valid_idx = stack[pair_depth]
                if pre_valid_idx == first_prod_idx:
                    chain = chain_for_index[first_prod_idx]
                    if chain is not None:
                        pre_valid_idx = chain[0]
                pre_key = (pair_id, pre_valid_idx)
                if pre_key not in seen_pre_valid:
                    seen_pre_valid.add(pre_key)
                    insert_before.setdefault(pre_valid_idx, []).append(
                        Instruction("event_set", event=_event_for(seg_id, pair_id, 0))
                    )
            ready_key = (pair_id, idx)
            if ready_key not in seen_ready:
                seen_ready.add(ready_key)
                insert_before.setdefault(idx, []).append(
                    Instruction("event_set", event=_event_for(seg_id, pair_id, 1))
                )
                insert_before.setdefault(idx, []).append(
                    Instruction("event_wait", event=_event_for(seg_id, pair_id, 1))
                )

    insert_valid_after: Dict[int, List[Instruction]] = {}
    seen_valid: Set[Tuple[int, int]] = set()
    seen_valid_pairs: Set[Tuple[int, int]] = set()
    consumer_pairs: Set[Tuple[int, int]] = set()
    loop_stack = []

    for idx, inst in enumerate(insts):
        if inst.opname == "start_loop" and in_segment[idx] and seg_id_per_index[idx] != -1:
            loop_stack.append(idx)
        elif inst.opname == "end_loop" and loop_stack and in_segment[idx] and seg_id_per_index[idx] != -1:
            loop_stack.pop()

        if not in_segment[idx] or seg_id_per_index[idx] == -1 or inst.opname not in consumer_ops:
            continue
        seg_id = seg_id_per_index[idx]
        depths = segment_depths.get(seg_id, [])
        consumer_depth = depth_per_index[idx]
        if not depths:
            depths = [0]
        if name_level == "fix":
            desired_depths = []
            for pid, depth in enumerate(depths):
                desired = depth
                min_depth = min_consumer_depth.get((seg_id, pid))
                if min_depth is not None and min_depth < desired:
                    desired = min_depth
                desired_depths.append(desired)
            eligible_pairs = [
                pid for pid, desired in enumerate(desired_depths) if desired <= consumer_depth
            ]
        elif name_level in ("l1", "ubin", "ubout"):
            eligible_pairs = [pid for pid, depth in enumerate(depths) if depth <= consumer_depth]
            if name_level in ("ubin", "ubout") and not eligible_pairs:
                eligible_pairs = [pair_id_per_index[idx]]
        else:
            eligible_pairs = [pair_id_per_index[idx]]

        for pair_id in eligible_pairs:
            if name_level == "ubin":
                last_idx = last_consumer_idx.get((seg_id, pair_id))
                if last_idx is None or last_idx != idx:
                    continue
            if name_level == "fix":
                pair_depth = desired_depths[pair_id] if pair_id < len(desired_depths) else 0
            else:
                pair_depth = depths[pair_id] if pair_id < len(depths) else 0
            consumer_pairs.add((seg_id, pair_id))
            if name_level in ("l1", "fix", "ubin", "ubout"):
                if consumer_depth > pair_depth:
                    if len(loop_stack) > pair_depth:
                        start_at_depth = loop_stack[pair_depth]
                        end_idx = loop_end_for_start.get(start_at_depth)
                        if end_idx is not None:
                            key = (pair_id, end_idx)
                            if key not in seen_valid:
                                seen_valid.add(key)
                                insert_valid_after.setdefault(end_idx, []).append(
                                    Instruction("event_set", event=_event_for(seg_id, pair_id, 0))
                                )
                            continue
            else:
                if consumer_depth > pair_depth and consumer_depth > 0 and len(loop_stack) >= consumer_depth:
                    start_at_depth = loop_stack[consumer_depth - 1]
                    end_idx = loop_end_for_start.get(start_at_depth)
                    if end_idx is not None:
                        key = (pair_id, end_idx)
                        if key not in seen_valid:
                            seen_valid.add(key)
                            insert_valid_after.setdefault(end_idx, []).append(
                                Instruction("event_set", event=_event_for(seg_id, pair_id, 0))
                            )
                    continue

            next_idx = _next_significant_index(idx)
            if name_level == "fix":
                same_pair_next = (
                    next_idx != -1
                    and insts[next_idx].opname in consumer_ops
                    and depth_per_index[next_idx] >= pair_depth
                )
            elif name_level in ("l1", "ubin", "ubout"):
                same_pair_next = (
                    next_idx != -1
                    and insts[next_idx].opname in consumer_ops
                    and depth_per_index[next_idx] >= pair_depth
                )
            else:
                same_pair_next = (
                    next_idx != -1
                    and pair_id_per_index[next_idx] == pair_id
                    and insts[next_idx].opname in consumer_ops
                )
            if not same_pair_next:
                insert_idx = idx
                if name_level in ("l0", "fix"):
                    chain = chain_for_index[idx]
                    if chain is not None and seg_id_per_index[chain[1]] != -1:
                        insert_idx = chain[1]
                key = (pair_id, insert_idx)
                if key not in seen_valid:
                    seen_valid.add(key)
                    seen_valid_pairs.add((seg_id, pair_id))
                    insert_valid_after.setdefault(insert_idx, []).append(
                        Instruction("event_set", event=_event_for(seg_id, pair_id, 0))
                    )

    if name_level == "fix":
        for seg_pair in consumer_pairs:
            if seg_pair in seen_valid_pairs:
                continue
            seg_id, pair_id = seg_pair
            last_idx = last_producer_idx.get(seg_pair)
            if last_idx is None:
                continue
            insert_idx = last_idx
            chain = chain_for_index[last_idx]
            if chain is not None and seg_id_per_index[chain[1]] != -1:
                insert_idx = chain[1]
            depths = segment_depths.get(seg_id, [])
            pair_depth = depths[pair_id] if pair_id < len(depths) else 0
            consumer_depth = min_consumer_depth.get(seg_pair)
            if consumer_depth is not None and consumer_depth > pair_depth:
                stack = loop_stack_per_index[last_idx]
                if len(stack) > pair_depth:
                    outer_start = stack[pair_depth]
                    outer_end = loop_end_for_start.get(outer_start)
                    if outer_end is not None:
                        insert_idx = outer_end
            key = (pair_id, insert_idx)
            if key not in seen_valid:
                seen_valid.add(key)
                insert_valid_after.setdefault(insert_idx, []).append(
                    Instruction("event_set", event=_event_for(seg_id, pair_id, 0))
                )
    for idx, inst in enumerate(insts):
        if idx in insert_before:
            new_insts.extend(insert_before[idx])

        new_insts.append(inst)

        if idx in insert_after:
            new_insts.extend(insert_after[idx])

        if idx in insert_valid_after:
            new_insts.extend(insert_valid_after[idx])

    return new_insts


def insert_auto_sync(instructions: List[Instruction]) -> List[Instruction]:
    insts = list(instructions)
    if not insts:
        return insts
    insts = _insert_auto_sync_for_pipe(
        insts,
        _MTE2_OPS,
        _MTE1_OPS,
        Pipe.MTE2,
        Pipe.MTE1,
        "_tmp_event",
        "l1",
    )
    insts = _insert_auto_sync_for_pipe(
        insts,
        _MTE2_OPS,
        _get_v_ops(),
        Pipe.MTE2,
        Pipe.V,
        "_tmp_event",
        "ubin",
    )
    insts = _insert_auto_sync_for_pipe(
        insts,
        _get_v_ops(),
        _MTE3_OPS,
        Pipe.V,
        Pipe.MTE3,
        "_tmp_event",
        "ubout",
        consumer_sevent_predicate=_sevent_needed_for_consumer_src_only,
    )
    insts = _insert_auto_sync_for_pipe(
        insts,
        _MTE1_OPS,
        _M_OPS,
        Pipe.MTE1,
        Pipe.M,
        "_tmp_event",
        "l0",
    )
    insts = _insert_auto_sync_for_pipe(
        insts,
        _M_OPS,
        _FIX_OPS,
        Pipe.M,
        Pipe.FIX,
        "_tmp_event",
        "fix",
    )
    return insts
