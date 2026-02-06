# easyasc Summary

**Overview**
- `easyasc` is a Python DSL that records tensor/var operations as `Instruction`s and translates them into Ascend C++ code, split into cube (AIC) and vec (AIV) kernels.
- Typical flow: Python AST transforms normalize syntax, runtime objects append instructions, the parser splits cube/vec, inserts auto-sync, prunes, and emits C++ via handlers, and optional custom op project generation wraps the output.

**Repository Layout**
- `easyasc/`: Core library (DSL, runtime objects, parser, handlers, resources, shortcuts).
- `testcases/`: Example kernels covering vec ops, autosync patterns, cube/vec mixing, and syntax sugar.
- `test.py`: End-to-end sample kernel using auto-sync, matmul shortcut, mutex sync, and custom op + ACLNN test generation.
- `test_cust_op/`: Sample CustomOp build tree with CMake files and a helper build script.
- `legacy/`: Present but empty.
- `wsl_setenv.sh`: Environment setup for WSL (LD_LIBRARY_PATH, PYTHONPATH, include paths).
- `TODO.txt`: Note about merging tensorutils changes for 910b.

**Public API Surface (`easyasc/__init__.py`)**
- Re-exports datatypes (`DT`), tensors (`Tensor`, `DBuff`, `GMTensor`), `Var`, decorators (`kernel`, `func`, `auto_sync`), positions, pipe/events/mutex types, and a large set of stub functions.
- Re-exports vector helpers `maximum`/`minimum` (from `utils.vecop`) for syntax sugar.

**Decorators and Syntax Transforms**
- `decorators.py`: `kernel` and `func` wrap or transform functions using `pythonic.transform_kernel`, and `auto_sync` is a decorator/context manager that injects `start_auto_sync` and `end_auto_sync` instructions.
- `pythonic.py`: `_VarNameAdder` injects `name=` into `Var/Tensor/DBuff/...` calls and `range` iterators; `_BoolOpRewriter` converts `and/or/not` to `&/|/~`; `_IfRewriter` rewrites `if/elif/else` into `with If/Elif/Else` blocks; `transform_kernel` recompiles and preserves metadata.
- `flowcontrol.py`: `range/unroll` emit `start_loop`/`end_loop` instructions and validate args; `If/Elif/Else` are context managers that emit block instructions and enforce kernel-only usage.

**Kernel Lifecycle (`kernelbase/kernelbase.py`)**
- `KernelBase` stores kernel name, transformed function, instruction list, cross-core mutexes, and `workspace_shapes` for split workspaces.
- `KernelBase.__call__` binds args, names `GMTensor`/`Var` parameters, inserts `create_gm_tensor`, creates internal L0A/L0B buffers and counters for matmul shortcuts, emits `reset_cache`, runs the kernel, collects output `GMTensor`s, adds cross-core sync instructions, and resets global state.
- `dump_asc` emits `{path}_cube.h` and `{path}_vec.h` from the translator.
- `dump_kernel` emits cube/vec headers with `__aicore__ inline void {name}_{cube/vec}(...)` wrappers and a `.cpp` entry that loads tiling, dispatches AIC/AIV, and passes GM/workspace/var params.
- `generate_op_host` writes a tiling data header and op-def `.cpp` with inferred inputs/outputs/attrs, tiling size calculation (including `workspace_shapes`), and block size setup.
- `generate_op_project` extracts `resources/CustomOp.tar.gz` and updates `CMakePresets.json` for `ASCEND_CANN_PACKAGE_PATH` and device type.
- `generate` now accepts `cann_path: Optional[str]` (falling back to `ASCEND_HOME_PATH`) plus `profile: bool`, then generates the op project, kernel, ACLNN test at `{out_dir}_aclnn_test`, and helper scripts via `generate_bashfiles`.
- `generate_aclnn_test` creates an ACLNN test scaffold: copies `macros.h`/`parse_prof.py`, writes `setup_aclnn.py` with injected `cann_path` and architecture-specific `x86_64-linux`/`aarch64-linux` paths, ensures `input/` and `output/` subdirs exist, emits `tensorx.h` with `#include "aclnn_{name}.h"`, and generates `test.cpp` from the last bound kernel args/outputs (requires the kernel be called). It maps dtypes to `TensorX<...>` macros, enforces `Var` to be `int`/`float` with values, orders `EXECOP` args as inputs → vars → outputs, uses `aclnn{CamelName}`, and when `profile=True` inserts profiling blocks from `resources/test.cpp` and runs a 100-iteration loop plus a single call.
- `generate_bashfiles` writes `b.sh` (build/run/install + setup) and `r.sh` (runtime execution) helper scripts in the current directory using the provided `path` and `cann_path`.

**Runtime Data Model (`easyasc/utils`)**
- `Tensor`/`DBuff`/`GMTensor` in `utils/Tensor.py` validate types, allocate temp names, emit `create_*`, support slicing via `__getitem__`, and use `__ilshift__` to select the correct data-move operation by position.
- `GMTensor` supports up to 2 sliced dimensions and exposes `bind_cv_mutex/bind_vc_mutex` plus `lock/ready/wait/free` for cross-core sync.
- `Var` in `utils/var.py` records `create_var`, tracks dtype/value, supports arithmetic via stub functions, and returns `Expr` for comparisons (which intentionally fail Python boolean evaluation).
- `VecOP` in `utils/vecop.py` captures `dst <<= src1 + src2` style operations and emits the correct vector stub calls, reinterpreting int-only ops when needed.
- Type and enum helpers in `datatype.py`, `positions.py`, `pipe.py`, `comparemode.py`, `roundmode.py`, and `selectmode.py` provide small wrapper types and C++ mappings; `Datatype.int64` is now available.
- `events.py` defines `SEvent`/`DEvent` with `set/wait/setall/release` instruction emission.
- `mutex.py` defines `CvMutex`/`VcMutex` and registers them with the active kernel for cross-core coordination.

**Parser and Translation (`easyasc/parser`)**
- `asc.py` classifies instructions into cube/vec sides using stub opnames and handler modules, validates block structure, folds loops/decl-assigns, and emits C++ via handlers.
- `split_instructions` runs pruning passes per side, and `translate_split` inserts auto-sync and runs memory-usage analysis before translation.
- `analyze_usage` prints rich tables for L1/L0/UB usage estimates and inserts a reset-cache banner.
- `asc_utils.py` maps dtype/positions to C++ strings, formats expressions and offsets, simplifies expressions with SymPy when available, and tracks which temporary instructions can be skipped or inlined; it now substitutes `expr_map` entries inside `Expr` conditions so folded temp vars (e.g., `GetSubBlockIdx`) inline correctly in control flow.
- `asc_pruning.py` parses instructions into structured blocks, removes empty blocks, prunes unused `create_gm_tensor`/event declarations, and prunes unused vars/assignments using side-specific dependency analysis.
- `asc_autosync.py` builds pipe-to-op mappings (including explicit vec ops), uses `AutosyncNode` to walk nested loops/ifs and detect buffer reuse, and inserts event set/wait with `preset` hints when names contain `valid`.
- `asc_handlers/` provide opcode-to-emitter handlers for core creation/slicing, math ops, flow control, cube ops, vec ops, events, pipe barriers/flags, atomics, reinterpret, and reset-cache.

**Stub Functions (`easyasc/stub_functions`)**
- Thin Python APIs that validate inputs and append `Instruction`s for cube/vec ops.
- `var_op.py` includes `CeilDiv`, `Min/Max`, alignment helpers, scalar sqrt, and arithmetic with constant folding when possible.
- `cube.py`, `vec/`, `atomic.py`, `barrier.py`, `flags.py`, `crosscore.py`, and `misc.py` cover data movement, math, atomics, barriers, flag sync, cross-core ready/wait, reinterpret, split workspace, and cache reset.

**Shortcuts (`easyasc/shortcuts`)**
- `matmul.py` provides a matmul helper using internal L0A/L0B buffers, supports split-n/split-k, handles transpose-aware indexing, and optionally reinterprets int types for int4 paths.

**Resources**
- `resources/CustomOp.tar.gz` and `resources/CMakePresets.json` template a CANN custom op project.
- `resources/tensorutils.h` is bundled into generated kernel code.
- `resources/macros.h` defines `TensorX` dtype macros (including INT64/UINT8/UINT16/UINT32/UINT64 additions).
- `resources/test.cpp` serves as the profiling snippet template for ACLNN test generation.

**Tests and Examples**
- `test.py` demonstrates auto-sync, matmul shortcut, workspace splitting, mutex sync, cache reset, and custom op generation (`KernelBase.generate`), and now also exercises `generate_aclnn_test`.
- `testcases/test_allvec.py` exercises most vector ops, barriers, events, and atomic variants in one kernel.
- `testcases/test_cube_autosync.py` stresses autosync insertion across nested loops/ifs and mixed cube/vec scopes.
- `testcases/test_cvmix.py` mixes cube and vec phases with control flow and sync events.
- `testcases/test_syntaxsugar.py` and `testcases/test_vecop.py` validate `<<=` syntax sugar, `VecOP`, and scalar/vector overloads.
- `testcases/test_vec_autosync.py` focuses on vec-side autosync with workspace splitting and nested loops.
