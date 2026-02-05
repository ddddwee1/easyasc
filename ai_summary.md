# easyasc Summary

**Project Overview**
- `easyasc` is a Python front-end that records tensor/var operations as `Instruction`s and translates them into Ascend C++ code (cube/vec) through the parser + handler pipeline.
- Flow: decorators + AST transforms create `KernelBase`; runtime objects (`Var`, `Tensor`, `DBuff`, `GMTensor`, events/mutex) append instructions; `parser/asc.py` validates/simplifies, inserts auto-sync, and emits C++.

**easyasc/__init__.py**
- Re-exports core types, decorators, positions/pipes/events/mutex, and all stub ops (var/cube/vec/misc, including `split_workspace` and `reset_cache`).

**easyasc/decorators.py**
- `kernel` wraps a transformed function in `KernelBase`; if the function was `auto_sync`-decorated, it wraps with start/end markers.
- `func` applies AST transforms (and optional auto-sync wrapping) without creating a `KernelBase`.
- `auto_sync` is a decorator/context manager that emits `start_auto_sync`/`end_auto_sync` instructions around a call or `with` block.

**easyasc/flowcontrol.py**
- `range`/`unroll` create loop vars and emit `start_loop`/`end_loop`, validating args and non-zero step.
- `If/Elif/Else` context managers emit `start_if/start_elif/start_else/end_if` and enforce kernel-only usage.

**easyasc/globvars.py**
- Global state: `active_kernel`, `tmp_idx`, atomic settings, `device_type`, and memory cap constants (L1/L0/UB) used by usage analysis.

**easyasc/kernelbase/kernelbase.py**
- `KernelBase` stores the kernel name, transformed function, instruction list, and cross-core mutexes.
- `KernelBase` tracks `workspace_shapes` to record shapes from `split_workspace` during a call.
- `__call__` clears `workspace_shapes`, binds args, assigns names to `GMTensor`/`Var`, emits `create_gm_tensor`, runs the kernel, then inserts cross-core ready/wait instructions from `CvMutex`/`VcMutex` before resetting globals.
- `dump_asc` writes `{path}_cube.h` and `{path}_vec.h` via `translate_split`.
- `dump_kernel` wraps emitted code with `tensorutils.h`, `TPipe* pipe_ptr = GetTPipePtr();`, and `int _offset = 0;` inside `__aicore__ inline void {name}_{cube/vec}(...)`. It also emits `{path}.cpp` entry boilerplate with tiling extraction for `Var` params and `ASCEND_IS_AIC/AIV` dispatch.
- `generate_op_host` emits `{name}_tiling.h` describing tiling data fields for input `Var` parameters (CamelCase op + `TilingData`) and registers the tiling class.
- `generate_op_project(path, cann_path)` extracts `easyasc/resources/CustomOp.tar.gz` (skips if target dir exists and is non-empty) and writes `CMakePresets.json` from template, replacing `ASCEND_CANN_PACKAGE_PATH` and optional `ASCEND_COMPUTE_UNIT` based on `globvars.device_type` (`b*`→`ascend910b`, `d*`→`ascend910_93`).

**easyasc/pythonic.py**
- `_VarNameAdder` injects `name=` for assignment targets on calls like `Var`, `Tensor`, `DBuff`, `Min`, `CeilDiv`, `range`, `SEvent`, `DEvent`, `reinterpret`, and `split_workspace`.
- `_BoolOpRewriter` turns `and/or/not` into bitwise `&/|/~` for expression construction.
- `_IfRewriter` converts `if/elif/else` statements into `with If/Elif/Else` blocks.
- `transform_kernel` parses source, applies transforms, recompiles, and preserves metadata.

**easyasc/parser/asc.py**
- `translate` validates block structure, builds expression state, folds decl/loop patterns, and emits C++ via handlers; unhandled opnames are commented.
- `translate_split` splits cube/vec instructions, inserts auto-sync, runs `analyze_usage` per side, then translates each side.
- `analyze_usage` prints `rich` tables grouped by `Position` for `create_tensor/create_dbuf`, estimating size from the first two shape dims and `dtype.size` (DBuff doubled). It prints per-position `Usage: used KB / cap KB` and restarts after `reset_cache` with a centered reset banner and optional centered label line (`{name}_cube/vec`).

**easyasc/parser/asc_autosync.py**
- Builds pipe/opname maps and inserts auto-sync event instructions between producer/consumer segments.
- `AutosyncNode` analyzes pipe usage and buffer reuse; auto-created events set `preset=True` for names containing `valid` and swap src/dst pipes accordingly.

**easyasc/parser/asc_utils.py**
- dtype/position C++ mapping, expression simplification, and value-to-C++ formatting.
- Tracks assignment-style ops and temporary expression maps used by `translate`.

**easyasc/parser/asc_pruning.py**
- `prune_empty_blocks`, `prune_unused_decls`, and `prune_unused_vars` remove dead loops/ifs and unused create_* / create_var declarations.

**easyasc/parser/helper.py**
- `CodeHelper` is an indentation-aware string builder for emitted C++ code.

**easyasc/parser/asc_handlers/**
- Handler registry maps `Instruction.opname` to C++ emitters.
- `core.py`: `create_var/dbuf/tensor/gm_tensor`, `split_workspace`, `get_buf`, slice helpers; `create_tensor`/`create_dbuf` compute `numel` from shape expressions (DBuff calls `.Init(numel)`).
- `cube.py`: GM<->L1/L0, MMAD, L0C->GM, and `l0c_to_l1` (`L0C2L1`).
- `vec_*`: vector arithmetic, unary, scalar, compare/select, data movement, gather/scatter, sort, group, cast, vecmask.
- `events.py`: `SEvent/DEvent` creation with `preset` template arg plus set/wait/release operations.
- `pipe_ops.py`: PIPE barrier/ready/wait emitters.
- `math_ops.py`: scalar ops (`GetCubeNum/Idx`, `CeilDiv`, `Min/Max`, arithmetic).
- `flow.py`: loop and if/else emitters.
- `reinterpret.py`: tensor reinterpret cast.
- `misc.py`: `reset_cache` emits `pipe_ptr->Reset();` and `OccupyMMTE1Events();`.

**easyasc/stub_functions/**
- Thin Python APIs that validate inputs and append `Instruction`s.
- `var_op.py`: scalar math with dtype inference and constant folding for numeric inputs; includes `Align16/32/64/128/256`, `scalar_sqrt`, `GetVecNum/Idx`, `GetSubBlockIdx`.
- `cube.py`: cube/matrix ops (`gm_to_l1_nd2nz`, `l1_to_l0`, `mmad`, `l0c_to_gm_nz2nd`, `l0c_to_l1`).
- `flags.py`: `setflag`/`waitflag` pipe event instructions.
- `vec/`: vector ops by category (binary/unary/unaryscalar/group/compare/select/sort/datamove/cast/gather/scatter/vecmask).
- `atomic.py`: atomic add/max/min and end.
- `barrier.py`: pipe barriers per pipe.
- `crosscore.py`: cube/vec ready/wait sync ops.
- `misc.py`: `reinterpret`, `split_workspace` (creates `GMTensor` from workspace and records its shape on the active kernel), and `reset_cache`.

**easyasc/utils/**
- `datatype.py`: `DataTypeValue` plus `Datatype` definitions.
- `var.py`: `Var` tracks dtype/value/name, emits `create_var`, and forwards arithmetic/comparison to `stub_functions.var_op`.
- `Tensor.py`: `Tensor`, `DBuff`, `GMTensor` classes; creation instructions, slicing/views, vector op overloads, GM binding, and mutex helpers; `Tensor <<= L0C` uses `l0c_to_l1`.
- `positions.py`: `Position` enum mapping to C++.
- `pipe.py`: `Pipe` enum and `PipeType` wrappers.
- `events.py`: `SEvent/DEvent` with `preset` flag and event ops.
- `mutex.py`: `CvMutex`/`VcMutex` register with the active kernel for cross-core sync.
- `instruction.py`: `Instruction` container (opname + kwargs).
- `vecop.py`: deferred vector op wrapper consumed by `<<=`, including reinterpret for int-only ops when needed.
- `comparemode.py`, `roundmode.py`, `selectmode.py`: enum-like wrappers for modes.
