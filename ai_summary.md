# easyasc Summary

**Overview**
- `easyasc` is a Python DSL that records tensor/var operations as `Instruction`s and translates them into Ascend C++ code, split into cube (AIC) and vec (AIV) kernels.
- Typical flow: Python AST transforms normalize syntax, runtime objects append instructions, the parser splits cube/vec, inserts auto-sync, prunes, and emits C++ via handlers, and optional custom op project generation wraps the output.
- `OpExec` provides a PyTorch-facing wrapper that builds `GMTensor`/`Var` placeholders from torch inputs, triggers code/test generation, and can run the build script.

**Repository Layout**
- `easyasc/`: Core library (DSL, runtime objects, parser, handlers, resources, shortcuts).
- `easyasc/torchplutin.py`: PyTorch-facing execution wrapper and build runner (`OpExec`).
- `testcases/`: Example kernels covering vec ops, autosync patterns, cube/vec mixing, and syntax sugar.
- `test.py`: End-to-end sample kernel using auto-sync, matmul shortcut, mutex sync, and custom op + ACLNN test generation.
- `legacy/`: Present but empty.
- `wsl_setenv.sh`: Environment setup for WSL (LD_LIBRARY_PATH, PYTHONPATH, include paths).
- `TODO.txt`: Note about merging tensorutils changes for 910b.

**Public API Surface (`easyasc/__init__.py`)**
- Re-exports datatypes (`DT`), tensors (`Tensor`, `DBuff`, `GMTensor`), `Var`, decorators (`kernel`, `func`, `auto_sync`), positions, pipe/events/mutex types, and a large set of stub functions.
- Re-exports `OpExec` as the PyTorch execution helper.
- Re-exports vector helpers `maximum`/`minimum` (from `utils.vecop`) for syntax sugar.

**Decorators and Syntax Transforms**
- `decorators.py`: `kernel` and `func` wrap or transform functions using `pythonic.transform_kernel`, and `auto_sync` is a decorator/context manager that injects `start_auto_sync` and `end_auto_sync` instructions.
- `pythonic.py`: `_VarNameAdder` injects `name=` into `Var/Tensor/DBuff/Reg/MaskReg/CastConfig/...` calls and `range` iterators; `_BoolOpRewriter` converts `and/or/not` to `&/|/~`; `_IfRewriter` rewrites `if/elif/else` into `with If/Elif/Else` blocks; `transform_kernel` uses `inspect.getsourcelines` plus `ast.increment_lineno` so transformed traceback line numbers match original source files.
- `flowcontrol.py`: `range/unroll` validate args and emit loop instructions to the active context (`start_loop` in kernel scope, `start_micro_loop` in micro scope, both closed by `end_loop`); `If/Elif/Else` emit branch markers to the same active target (kernel or micro).

**Kernel Lifecycle (`kernelbase/kernelbase.py`)**
- `KernelBase` stores kernel name, transformed function, instruction list, cross-core mutexes, and `workspace_shapes` for split workspaces.
- `KernelBase.__call__` binds args, names `GMTensor`/`Var` parameters, inserts `create_gm_tensor`, creates internal L0A/L0B buffers and counters for matmul shortcuts, emits `reset_cache`, runs the kernel, collects output `GMTensor`s, adds cross-core sync instructions, and resets global state.
- `dump_asc` emits `{path}_cube.h` and `{path}_vec.h` from the translator.
- `dump_kernel` emits cube/vec headers with `__aicore__ inline void {name}_{cube/vec}(...)` wrappers and a `.cpp` entry that loads tiling, dispatches AIC/AIV, and passes GM/workspace/var params.
- `generate_op_host` writes a tiling data header and op-def `.cpp` with inferred inputs/outputs/attrs, tiling size calculation (including `workspace_shapes`), and block size setup.
- `generate_op_project` extracts `resources/CustomOp.tar.gz` and updates `CMakePresets.json` for `ASCEND_CANN_PACKAGE_PATH` and device type.
- `generate` accepts `out_dir: str = ""` (empty defaults to kernel name), `cann_path: Optional[str]` (falling back to `ASCEND_HOME_PATH`) and `profile: bool`, then generates the op project, kernel, ACLNN test at `{out_dir}_aclnn_test`, and helper scripts via `generate_bashfiles`.
- `generate_aclnn_test` creates an ACLNN test scaffold: copies `macros.h`/`parse_prof.py`, writes `setup_aclnn.py` with injected `cann_path` and architecture-specific `x86_64-linux`/`aarch64-linux` paths, ensures `input/` and `output/` subdirs exist, emits `tensorx.h` with `#include "aclnn_{name}.h"`, and generates `test.cpp` from the last bound kernel args/outputs (requires the kernel be called). It maps dtypes to `TensorX<...>` macros, enforces `Var` to be `int`/`float` with values, orders `EXECOP` args as inputs → vars → outputs, uses `aclnn{CamelName}`, and when `profile=True` inserts profiling blocks from `resources/test.cpp` and runs a 100-iteration loop plus a single call.
- `generate_bashfiles` writes `b.sh` (build/run/install + setup) and `r.sh` (runtime execution) helper scripts in the current directory using the provided `path` and `cann_path`.

**Torch Integration (`easyasc/torchplutin.py`)**
- `_run_bash_with_progress` runs `b.sh`, prints a simple progress bar to stderr, logs stdout/stderr to `b.sh.log`, and raises on nonzero exit.
- `OpExec` validates a `KernelBase` instance, requires all `torch.Tensor` args to precede scalar args, maps torch dtypes to `Datatype`, builds `GMTensor`/`Var` placeholders (reusing scalar vars for shape dims), runs the kernel, calls `KernelBase.generate`, writes `input_{name}.bin` into `{out_dir or kernel_name}_aclnn_test/input`, and optionally runs `b.sh` unless `gen_only=True`.

**Runtime Data Model (`easyasc/utils`)**
- `Tensor`/`DBuff`/`GMTensor` in `utils/Tensor.py` validate types, allocate temp names, emit `create_*`, support slicing via `__getitem__`, and use `__ilshift__` to select the correct data-move operation by position. `Tensor` now accepts `Reg`/`RegList`/`RegOP` stores to UB; for `RegList` stores, it emits lane-wise writes using block stride `256 // dtype.size`. It also supports scalar shorthand indexing (`x[i]`) for last-dimension offset updates, carries `vec_copy_mode` plus helpers (`downsample/upsample/unpack/unpack4/brcb/single`) to influence UB->Reg loads, and emits `micro_slice_tensor` when slicing inside `active_micro`.
- `GMTensor` supports up to 2 sliced dimensions and exposes `bind_cv_mutex/bind_vc_mutex` plus `lock/ready/wait/free` for cross-core sync.
- `Var` in `utils/var.py` records `create_var`, tracks dtype/value, supports arithmetic via stub functions, and returns `Expr` for comparisons (which intentionally fail Python boolean evaluation).
- `Reg`/`MaskReg` in `utils/reg.py` now expose RegOP-based operators, casting helpers, mask ops, and `<<=` assignment that can emit micro ops or perform UB<->Reg moves via RegOP, including dtype-mismatch Tensor assignment via temp-reg cast path.
- `RegOP` in `utils/regop.py` records micro operations and materializes them through `<<=`; it covers arithmetic, unary/scalar, group, cast, compare/select, mask ops (including interleave/deinterleave), and datamove/gather/scatter variants.
- `VecOP` in `utils/vecop.py` captures `dst <<= src1 + src2` style operations and emits the correct vector stub calls, reinterpreting int-only ops when needed.
- `micro/micromodule.py`: `MicroModule` tracks input arguments for codegen, caches a default cast config, collects `cast_cfg_list`, and `gen_code` emits a full micro header (pragma/include, `CastTrait` definitions, `__aicore__` wrapper, and `__VEC_SCOPE__` around translated instructions).
- `CastConfig` (in `utils/castconfig.py`) carries `round_mode`, `reg_layout`, and saturation settings; it auto-registers in the active micro’s `cast_cfg_list`, and pythonic auto-name injection supplies `name=...` when assigned.
- Type and enum helpers in `datatype.py`, `positions.py`, `pipe.py`, `comparemode.py`, `roundmode.py`, and `selectmode.py` provide small wrapper types and C++ mappings; `Datatype.int64` is now available.
- `events.py` defines `SEvent`/`DEvent` with `set/wait/setall/release` instruction emission.
- `mutex.py` defines `CvMutex`/`VcMutex` and registers them with the active kernel for cross-core coordination.

**Parser and Translation (`easyasc/parser`)**
- `asc.py` classifies instructions into cube/vec sides using stub opnames and handler modules, validates block structure, folds loops/decl-assigns, supports micro-loop starts (`start_micro_loop`) in loop-depth validation/folding, and emits C++ via handlers.
- `split_instructions` runs pruning passes per side, and `translate_split` inserts auto-sync and runs memory-usage analysis before translation.
- `analyze_usage` prints rich tables for L1/L0/UB usage estimates and inserts a reset-cache banner.
- `asc_utils.py` maps dtype/positions to C++ strings, formats expressions and offsets, simplifies expressions with SymPy when available, and tracks which temporary instructions can be skipped or inlined; it substitutes `expr_map` entries inside `Expr` conditions so folded temp vars (e.g., `GetSubBlockIdx`) inline correctly in control flow, folds `micro_slice_tensor` into pointer-arithmetic expressions (so generated micro code can avoid `_tmp_tensor` declarations), and drops zero-multiplied terms when building linear offsets.
- `asc_pruning.py` parses instructions into structured blocks, removes empty blocks, prunes unused `create_gm_tensor`/event declarations, structurally recognizes `start_micro_loop` as a loop block marker, and prunes unused vars/assignments using side-specific dependency analysis; dependency propagation now treats `micro_slice_tensor` like other tensor view defs/sources.
- `asc_autosync.py` builds pipe-to-op mappings (including explicit vec ops), uses `AutosyncNode` to walk nested loops/ifs and detect buffer reuse, and inserts event set/wait with `preset` hints when names contain `valid`.
- `asc_handlers/` provide opcode-to-emitter handlers for core creation/slicing, math ops, flow control (including `start_micro_loop` loop syntax), cube ops, vec ops, events, pipe barriers/flags, atomics, reinterpret, and reset-cache; core handlers now include `micro_slice_tensor` emission as `__ubuf__ {dtype}* out = src + offset`.

**Stub Functions (`easyasc/stub_functions`)**
- Thin Python APIs that validate inputs and append `Instruction`s for cube/vec ops.
- `var_op.py` includes `CeilDiv`, `Min/Max`, alignment helpers, scalar sqrt, and arithmetic with constant folding when possible; var-op instructions now route to `active_micro` first when inside micro scope.
- `cube.py`, `vec/`, `atomic.py`, `barrier.py`, `flags.py`, `crosscore.py`, and `misc.py` cover data movement, math, atomics, barriers, flag sync, cross-core ready/wait, reinterpret, split workspace, and cache reset.

**Shortcuts (`easyasc/shortcuts`)**
- `matmul.py` provides a matmul helper using internal L0A/L0B buffers, supports split-n/split-k, handles transpose-aware indexing, and optionally reinterprets int types for int4 paths.

**Resources**
- `resources/CustomOp.tar.gz` and `resources/CMakePresets.json` template a CANN custom op project.
- `resources/tensorutils.h` is bundled into generated kernel code.
- `resources/macros.h` defines `TensorX` dtype macros (including INT64/UINT8/UINT16/UINT32/UINT64 additions).
- `resources/test.cpp` serves as the profiling snippet template for ACLNN test generation.

**Tests and Examples**
- `test.py` demonstrates auto-sync, matmul shortcut, workspace splitting, mutex sync, cache reset, and custom op generation (`KernelBase.generate`), while `addmic` provides RegList/RegOP `<<=` coverage including binary, unary, unary-scalar, reduce, and `Tensor <<= RegList` paths; generated `addmic.h` now folds micro slice views into inline pointer arithmetic without `_tmp_tensor` temporaries.
- `testcases/test_allvec.py` exercises most vector ops, barriers, events, and atomic variants in one kernel.
- `testcases/test_cube_autosync.py` stresses autosync insertion across nested loops/ifs and mixed cube/vec scopes.
- `testcases/test_cvmix.py` mixes cube and vec phases with control flow and sync events.
- `testcases/test_syntaxsugar.py` and `testcases/test_vecop.py` validate `<<=` syntax sugar, `VecOP`, and scalar/vector overloads.
- `testcases/test_vec_autosync.py` focuses on vec-side autosync with workspace splitting and nested loops.
