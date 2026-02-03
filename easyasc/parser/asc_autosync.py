from typing import Dict, List, Optional, Set, Tuple, Literal, Union
from functools import lru_cache

from .asc_utils import build_expr_state, should_skip_inst
from ..utils.events import DEvent, SEvent
from .asc_handlers import build_handlers
from ..utils.Tensor import Tensor
from ..utils.instruction import Instruction
from ..utils.pipe import Pipe, PipeType


_AUTO_SYNC_START = "start_auto_sync"
_AUTO_SYNC_END = "end_auto_sync"
_EXPLICIT_VEC_OPNAMES = {
    "abs",
    "add",
    "adds",
    "axpy",
    "brcb",
    "cadd",
    "cast",
    "cgadd",
    "cgmax",
    "cgmin",
    "cmax",
    "cmin",
    "compare",
    "compare_scalar",
    "cpadd",
    "div",
    "dup",
    "exp",
    "gather",
    "gm_to_ub_pad",
    "ln",
    "lrelu",
    "mergesort4",
    "mergesort_2seq",
    "mul",
    "muladddst",
    "muls",
    "rec",
    "relu",
    "reset_mask",
    "rsqrt",
    "scatter",
    "select",
    "set_atomic_type",
    "set_cmpmask",
    "set_mask",
    "sort32",
    "sqrt",
    "sub",
    "ub_to_gm_pad",
    "ub_to_ub",
    "vand",
    "vmax",
    "vmaxs",
    "vmin",
    "vmins",
    "vnot",
    "vor",
}


@lru_cache(maxsize=1)
def get_pipe_opnames() -> Dict[str, Set[str]]:
    pipe_opnames: Dict[str, Set[str]] = {
        str(Pipe.MTE2): {"gm_to_l1_nd2nz", "gm_to_ub_pad"},
        str(Pipe.MTE1): {"l1_to_l0"},
        str(Pipe.M): {"mmad"},
        str(Pipe.FIX): {"l0c_to_gm_nz2nd"},
        str(Pipe.MTE3): {"ub_to_gm_pad"},
    }
    vec_pipe_ops = set(_EXPLICIT_VEC_OPNAMES)
    for opnames in pipe_opnames.values():
        vec_pipe_ops -= opnames
    pipe_opnames[str(Pipe.V)] = vec_pipe_ops
    return pipe_opnames


PIPE_OPNAMES = get_pipe_opnames()
_PIPE_NAME_TO_TYPE: Dict[str, PipeType] = {
    str(Pipe.MTE2): Pipe.MTE2,
    str(Pipe.MTE1): Pipe.MTE1,
    str(Pipe.M): Pipe.M,
    str(Pipe.FIX): Pipe.FIX,
    str(Pipe.MTE3): Pipe.MTE3,
    str(Pipe.V): Pipe.V,
}
_OPNAME_TO_PIPE: Dict[str, PipeType] = {
    opname: _PIPE_NAME_TO_TYPE[pipe_name]
    for pipe_name, opnames in PIPE_OPNAMES.items()
    if pipe_name in _PIPE_NAME_TO_TYPE
    for opname in opnames
}
_EVENT_TYPES = (SEvent, DEvent)


class AutosyncNode:
    def __init__(self, instructions: List[Instruction], src_pipe: PipeType, dst_pipe: PipeType,
                             producer_src: bool, producer_dst: bool, consumer_src: bool, consumer_dst: bool, 
                             buf_name: str):
        self.insts = list(instructions)
        self.new_insts: List[Union[Instruction, "AutosyncNode"]] = []
        self.src_pipe = src_pipe
        self.dst_pipe = dst_pipe
        self.producer_src = producer_src
        self.producer_dst = producer_dst
        self.consumer_src = consumer_src
        self.consumer_dst = consumer_dst
        self.buf_idx = 0
        self.buf_name = buf_name
        self.events_to_be_created: List[str] = []
        self.has_inst = False
        self.is_mixed_scope = False
        self.is_single_buffer = False 
        self.used_pipes: Set[PipeType] = set()
        self.pre_process()
        self.summarize_used_pipes()

    def assign_buf_indices(self):
        for inst in self.new_insts:
            if isinstance(inst, AutosyncNode):
                if self.is_mixed_scope:
                    inst.buf_idx = self.buf_idx + 1 
                else:
                    inst.buf_idx = self.buf_idx
                inst.assign_buf_indices()
        return self 

    def summarize_used_pipes(self):
        used = self.used_pipes
        op_to_pipe = _OPNAME_TO_PIPE
        src_pipe = self.src_pipe
        dst_pipe = self.dst_pipe
        producer_src = self.producer_src
        producer_dst = self.producer_dst
        consumer_src = self.consumer_src
        consumer_dst = self.consumer_dst
        is_single_buffer = self.is_single_buffer
        for inst in self.new_insts:
            if isinstance(inst, AutosyncNode):
                if inst.used_pipes:
                    used.update(inst.used_pipes)
                if inst.is_single_buffer:
                    is_single_buffer = True
                continue
            if not isinstance(inst, Instruction):
                continue
            pipe = op_to_pipe.get(inst.opname)
            if pipe is not None and pipe in [self.src_pipe, self.dst_pipe]:
                used.add(pipe)
                self.has_inst = True
                if pipe == src_pipe:
                    if producer_src:
                        src = inst.kwargs.get("src")
                        if isinstance(src, Tensor):
                            source_buf = src.source_buf
                            if source_buf is None or isinstance(source_buf, Tensor):
                                is_single_buffer = True
                    if not is_single_buffer and producer_dst:
                        dst = inst.kwargs.get("dst")
                        if isinstance(dst, Tensor):
                            source_buf = dst.source_buf
                            if source_buf is None or isinstance(source_buf, Tensor):
                                is_single_buffer = True
                elif pipe == dst_pipe:
                    if consumer_src:
                        src = inst.kwargs.get("src")
                        if isinstance(src, Tensor):
                            source_buf = src.source_buf
                            if source_buf is None or isinstance(source_buf, Tensor):
                                is_single_buffer = True
                    if not is_single_buffer and consumer_dst:
                        dst = inst.kwargs.get("dst")
                        if isinstance(dst, Tensor):
                            source_buf = dst.source_buf
                            if source_buf is None or isinstance(source_buf, Tensor):
                                is_single_buffer = True
        self.is_single_buffer = is_single_buffer
        self.is_mixed_scope = len(used)==2 and self.has_inst

    def pre_process(self):
        insts = self.insts
        new_insts = self.new_insts
        n = len(insts)
        if n == 0:
            return
        start_loop = "start_loop"
        end_loop = "end_loop"
        if_starts = ("start_if", "start_elif", "start_else")
        end_if = "end_if"

        def _find_loop_end(idx: int) -> Optional[int]:
            depth = 1
            while idx < n:
                op = insts[idx].opname
                if op == start_loop:
                    depth += 1
                elif op == end_loop:
                    depth -= 1
                    if depth == 0:
                        return idx
                idx += 1
            return None

        def _find_if_end(idx: int) -> Optional[int]:
            depth = 1
            while idx < n:
                op = insts[idx].opname
                if op in if_starts:
                    depth += 1
                elif op == end_if:
                    depth -= 1
                    if depth == 0:
                        return idx
                idx += 1
            return None

        i = 0
        while i < n:
            if i==0 and (insts[i].opname==start_loop or insts[i].opname in if_starts):
                new_insts.append(insts[i])
                i += 1
                continue
            inst = insts[i]
            op = inst.opname
            if op == start_loop:
                end_idx = _find_loop_end(i + 1)
                if end_idx is None:
                    new_insts.extend(insts[i:])
                    break
                body = insts[i:end_idx + 1]
                sub_insts = AutosyncNode(
                    body,
                    self.src_pipe,
                    self.dst_pipe,
                    self.producer_src,
                    self.producer_dst,
                    self.consumer_src,
                    self.consumer_dst,
                    self.buf_name,
                )
                new_insts.append(sub_insts)
                i = end_idx + 1
                continue
            if op in if_starts:
                end_idx = _find_if_end(i + 1)
                if end_idx is None:
                    new_insts.extend(insts[i:])
                    break
                body = insts[i:end_idx + 1]
                sub_insts = AutosyncNode(
                    body,
                    self.src_pipe,
                    self.dst_pipe,
                    self.producer_src,
                    self.producer_dst,
                    self.consumer_src,
                    self.consumer_dst,
                    self.buf_name,
                )
                new_insts.append(sub_insts)
                i = end_idx + 1
                continue
            new_insts.append(inst)
            i += 1
    
    def insert_auto_sync_inst(self):
        if not self.is_mixed_scope:
            return self
        
        if self.is_single_buffer:
            event_valid: Union[SEvent, DEvent] = object.__new__(SEvent)
            event_valid.name = f"_tmp_sevent_valid_{self.buf_name}_{self.buf_idx}"
            event_valid.src_pipe = self.src_pipe
            event_valid.dst_pipe = self.dst_pipe
            event_valid.idx = 9999
            self.events_to_be_created.append(event_valid.name)
            event_ready: Union[SEvent, DEvent] = object.__new__(SEvent)
            event_ready.name = f"_tmp_sevent_ready_{self.buf_name}_{self.buf_idx}"
            event_ready.src_pipe = self.src_pipe
            event_ready.dst_pipe = self.dst_pipe
            event_ready.idx = 9998
            self.events_to_be_created.append(event_ready.name)
        else:
            event_valid: Union[SEvent, DEvent] = object.__new__(DEvent)
            event_valid.name = f"_tmp_devent_valid_{self.buf_name}_{self.buf_idx}"
            event_valid.src_pipe = self.src_pipe
            event_valid.dst_pipe = self.dst_pipe
            event_valid.idx = 9999
            self.events_to_be_created.append(event_valid.name)
            event_ready: Union[SEvent, DEvent] = object.__new__(DEvent)
            event_ready.name = f"_tmp_devent_ready_{self.buf_name}_{self.buf_idx}"
            event_ready.src_pipe = self.src_pipe
            event_ready.dst_pipe = self.dst_pipe
            event_ready.idx = 9998
            self.events_to_be_created.append(event_ready.name)

        valid_wait_inst = Instruction(opname="event_wait", event=event_valid)
        valid_set_inst = Instruction(opname="event_set", event=event_valid)
        ready_wait_inst = Instruction(opname="event_wait", event=event_ready)
        ready_set_inst = Instruction(opname="event_set", event=event_ready)


        declared = False 
        changed = False
        result: List[Instruction | AutosyncNode] = []
        for i in self.new_insts:
            if isinstance(i, AutosyncNode):
                if self.src_pipe in i.used_pipes and not declared:
                    result.append(valid_wait_inst)
                    declared = True
                elif self.dst_pipe in i.used_pipes and declared and not changed:
                    result.append(ready_set_inst)
                    result.append(ready_wait_inst)
                    declared = False
                    changed = True
                elif self.src_pipe in i.used_pipes and changed and not declared:
                    result.append(valid_set_inst)
                    declared = True
                    changed = False
                result.append(i)
            else:
                if not isinstance(i, Instruction):
                    raise TypeError("Expected Instruction or AutosyncNode")
                pipe = _OPNAME_TO_PIPE.get(i.opname)
                if pipe is not None:
                    if pipe == self.src_pipe and not declared:
                        result.append(valid_wait_inst)
                        declared = True
                    elif pipe == self.dst_pipe and declared and not changed:
                        result.append(ready_set_inst)
                        result.append(ready_wait_inst)
                        declared = False
                        changed = True
                    elif pipe == self.src_pipe and changed and not declared:
                        result.append(valid_set_inst)
                        declared = True
                        changed = False
                result.append(i)
            
        if changed:
            if result[-1].opname not in ['end_loop', 'end_if']:
                result.append(ready_set_inst)
            else:
                result.insert(-1, valid_set_inst)
        if declared:
            if result[-1].opname not in ['end_loop', 'end_if']:
                result.append(ready_set_inst)
                result.append(ready_wait_inst)
                result.append(valid_set_inst)
            else:
                result.insert(-1, ready_set_inst)
                result.insert(-1, ready_wait_inst)
                result.insert(-1, valid_set_inst)
            print('WARNING: NOT balanced auto_sync events, please check the code logic!')
        
        self.new_insts = result
        return self

    def get_instructions(self):
        result: List[Instruction | AutosyncNode] = []
        for i in self.new_insts:
            if isinstance(i, AutosyncNode):
                result.extend(i.insert_auto_sync_inst().get_instructions().new_insts)
                self.events_to_be_created.extend(i.events_to_be_created)
            else:
                result.append(i)
        # return result
        self.new_insts = result
        return self 

    def create_events(self, full_instructions: List[Instruction]):
        for name in set(self.events_to_be_created):
            if name.startswith("_tmp_sevent"):
                event: Union[SEvent, DEvent] = object.__new__(SEvent)
            else:
                event = object.__new__(DEvent)

            if 'valid' in name:
                event.src_pipe = self.dst_pipe
                event.dst_pipe = self.src_pipe
            else:
                event.src_pipe = self.src_pipe
                event.dst_pipe = self.dst_pipe
            event.name = name
            event.idx = 9999 if 'valid' in name else 9998

            if name.startswith("_tmp_sevent"):
                event_create_inst = Instruction(opname="create_sevent", val=event)
            else:
                event_create_inst = Instruction(opname="create_devent", val=event)
            full_instructions.append(event_create_inst)
        return self.new_insts


def _insert_autosync_node(instructions: List[Instruction], mode: Literal['cube', 'vec']) -> Tuple[List[Instruction], List[Instruction]]:
    event_creation: List[Instruction] = []
    if mode == 'vec':
        instructions = AutosyncNode(instructions, Pipe.MTE2, Pipe.V, False, True, False, False, 'ubin').assign_buf_indices().insert_auto_sync_inst().get_instructions().create_events(event_creation)
        instructions = AutosyncNode(instructions, Pipe.V, Pipe.MTE3, False, False, True, False, 'ubout').assign_buf_indices().insert_auto_sync_inst().get_instructions().create_events(event_creation)
    else:
        instructions = AutosyncNode(instructions, Pipe.MTE2, Pipe.MTE1, False, True, False, False, 'l1').assign_buf_indices().insert_auto_sync_inst().get_instructions().create_events(event_creation)
        instructions = AutosyncNode(instructions, Pipe.MTE1, Pipe.M, False, True, False, False, 'l0').assign_buf_indices().insert_auto_sync_inst().get_instructions().create_events(event_creation)
        instructions = AutosyncNode(instructions, Pipe.M, Pipe.FIX, False, False, True, False, 'fix').assign_buf_indices().insert_auto_sync_inst().get_instructions().create_events(event_creation)
    return instructions, event_creation


def insert_auto_sync(instructions: List[Instruction], mode: Literal['cube', 'vec']) -> List[Instruction]:
    if mode not in ['cube', 'vec']:
        raise ValueError("mode must be either 'cube' or 'vec'")
    
    insts = list(instructions)
    if not insts:
        return insts
    result: List[Instruction] = []
    tmp_insts: List[Instruction] = []
    curr_inst_list = result

    for i in insts:
        if i.opname==_AUTO_SYNC_START:
            curr_inst_list = tmp_insts
        curr_inst_list.append(i)
        if i.opname==_AUTO_SYNC_END:
            tmp_insts_with_autosync, event_creation = _insert_autosync_node(tmp_insts, mode)
            result.extend(tmp_insts_with_autosync)
            result = event_creation + result
            curr_inst_list = result
            tmp_insts = []
    return result
