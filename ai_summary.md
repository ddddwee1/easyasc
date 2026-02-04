# easyasc Summary

**Project Overview**
- `easyasc` is a Python front-end that builds an instruction list (`Instruction`) from high-level tensor/var operations, then translates it into C++-style ASC code via the parser/handlers pipeline.
- Core flow: user code (decorators + Pythonic transforms) -> runtime objects (`Var`, `Tensor`, `DBuff`, `GMTensor`, events/mutex) emit `Instruction`s -> `parser/asc.py` validates, simplifies, and emits code using handler mappings.

**easyasc/__init__.py**
- Public API aggregator: re-exports data types, tensor/var classes, decorators, positions/pipes/events/mutex, and all stub ops (var ops, cube ops, vec ops, misc ops like `split_workspace` and `reset_cache`).

**easyasc/decorators.py**
- Defines `@kernel`, `@func`, and `@auto_sync` decorators.
- `kernel` wraps a transformed function in `KernelBase`, handling auto-sync wrapper if needed.
- `func` applies AST transforms but does not create a `KernelBase`.
- `auto_sync` injects `start_auto_sync`/`end_auto_sync` instructions around a call or context manager block.

**easyasc/flowcontrol.py**
- Custom `range`/`unroll` for kernel loops emitting `start_loop`/`end_loop` instructions.
- `If/Elif/Else` context managers emit `start_if/start_elif/start_else/end_if` instructions, with validation that they run inside a kernel.

**easyasc/globvars.py**
- Global state: current kernel (`active_kernel`), temporary index counter (`tmp_idx`), atomic settings, and device type.

**easyasc/kernelbase/kernelbase.py**
- `KernelBase` stores the kernel function, instruction list, and cross-core mutex metadata.
- `__call__` binds args, assigns names to `GMTensor`/`Var`, emits `create_gm_tensor`, executes the kernel function, and injects cross-core sync instructions from mutexes.
- `dump_asc` writes raw translated cube/vec code.
- `dump_kernel` wraps translated code with `#include "tensorutils.h"` and generates `__aicore__ inline void {name}_{cube/vec}(...);` signatures with ordered parameters: `GMTensor` as `GM_ADDR {name}_`, then `GM_ADDR workspace`, then `Var` parameters with their C++ dtypes. Inserts `int _offset = 0;` as the first statement.

**easyasc/pythonic.py**
- AST transforms for “pythonic” kernel syntax:
- `_VarNameAdder` auto-injects `name=` keyword into `Var/Tensor/DBuff/Min/CeilDiv/range/...` calls based on assignment target.
- `_BoolOpRewriter` rewrites `and/or/not` into bitwise `&/|/~` to build expression objects.
- `_IfRewriter` converts `if/elif/else` into `with If/Elif/Else` blocks.
- `transform_kernel` applies transforms and re-compiles the function while preserving metadata.

**easyasc/parser/asc.py**
- Core translation pipeline: validates block structure, builds expression state, and emits C++ statements via handlers.
- Supports folding of var declarations and expressions for temporary vars/tensors.
- `translate_split` splits instructions into cube/vec, inserts auto-sync, and translates each side.
- `analyze_usage` (used during translate_split) prints a centered, fixed-width rich table (width=50) grouped by `Position`, listing each `create_tensor/create_dbuf` `val` as `(Tensor|DBuff) name: size_kb` where size is `shape[0]*shape[1]/1024` if numeric.

**easyasc/parser/asc_autosync.py**
- Builds pipe/opname maps and inserts auto-sync (event-based) instructions between producer/consumer pipelines.
- `AutosyncNode` analyzes instruction blocks, pipe usage, and buffer reuse to decide when to insert sync events.

**easyasc/parser/asc_utils.py**
- Utility helpers: dtype/position mapping to C++ tokens, expression simplification, and value->C++ string conversion.
- Tracks assignment-style ops and builds temporary expression maps for inlining.

**easyasc/parser/asc_pruning.py**
- Pruning passes to remove unused or empty blocks and declarations:
- `prune_empty_blocks`: removes loops/if-chains that emit no code.
- `prune_unused_decls`: removes unused `create_*` instructions for tensors/buffers/events.
- `prune_unused_vars`: removes unused `create_var` declarations based on expression use.

**easyasc/parser/helper.py**
- `CodeHelper`: string builder with indentation tracking for emitting C++ code.

**easyasc/parser/asc_handlers/**
- Handler registry maps `Instruction.opname` to C++ code emitters.
- `core.py`: create_var/dbuf/tensor/gm_tensor, split_workspace, get_buf, slice operations.
- `math_ops.py`: scalar ops `GetCubeNum/GetCubeIdx/CeilDiv/Min/Max` and var arithmetic.
- `flow.py`: loop and if/else emitters.
- `cube.py`: GM->L1, L1->L0, MMAD, L0C->GM operations.
- `vec_*`: vector arithmetic, unary ops, scalar ops, compare/select, data movement, gather/scatter, sort, group ops.
- `pipe_ops.py`: PIPE barrier/ready/wait op emitters.
- `events.py`: create/set/wait/release events.
- `reinterpret.py`: tensor reinterpret cast.
- `misc.py`: `reset_cache` emits `_pipe->Reset();`.

**easyasc/stub_functions/**
- Thin Python APIs that validate inputs and append `Instruction`s.
- `var_op.py`: scalar math (mul/div/add/sub/min/max/ceildiv). Dtype inference, optional constant folding by updating `out.value` when both operands are numeric and denominator non-zero.
- `cube.py`: cube/matrix ops (`gm_to_l1_nd2nz`, `l1_to_l0`, `mmad`, `l0c_to_gm_nz2nd`).
- `vec/`: vector ops grouped by category (binary/unary/unaryscalar/group/compare/select/sort/datamove/cast/gather/scatter/vecmask).
- `atomic.py`: emits atomic add/max/min and end.
- `barrier.py`: pipe barriers for each pipe.
- `crosscore.py`: cube/vec ready/wait sync ops.
- `misc.py`: `reinterpret`, `split_workspace` (creates `GMTensor` from workspace), and `reset_cache`.

**easyasc/utils/**
- `datatype.py`: `DataTypeValue` and `Datatype` enum-like definitions (half/float/int/etc.).
- `var.py`: `Var` holds dtype/value/name and emits `create_var`; supports arithmetic by forwarding to `stub_functions.var_op` and builds `Expr` for comparisons.
- `Tensor.py`: `Tensor`, `DBuff`, `GMTensor` classes; emit create instructions; slicing and buffer views; operator overloads for vector ops; GM binding and mutex helpers.
- `positions.py`: `Position` enum -> C++ position mapping.
- `pipe.py`: `Pipe` enum and `PipeType` wrappers.
- `events.py`: `SEvent/DEvent` emit event instructions for set/wait/release.
- `mutex.py`: `CvMutex/VcMutex` use cross-core ops to coordinate cube/vec pipelines; auto-registered on active kernel.
- `instruction.py`: lightweight container for opname + kwargs.
- `vecop.py`: deferred vector op wrapper consumed by `<<=`; handles binary/unary/scalar/group/cast ops and does reinterpret for int-only ops when needed.
- `comparemode.py`, `roundmode.py`, `selectmode.py`: enum-like wrappers for modes used by vector compare/select/cast.
