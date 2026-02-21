# easyasc Architecture Summary (`easyasc/` only)

## 1. Scope and Snapshot
- This document summarizes only the `easyasc/` directory.
- Current snapshot (excluding `__pycache__`):
  - Total files: 107
  - Python files: 101
  - Python source lines: 15784
- Python line distribution by directory:
  - `easyasc/`: 7 files, 922 lines
  - `easyasc/kernelbase/`: 1 file, 943 lines
  - `easyasc/micro/`: 1 file, 180 lines
  - `easyasc/parser/`: 5 files, 2083 lines
  - `easyasc/parser/asc_handlers/`: 25 files, 2231 lines
  - `easyasc/shortcuts/`: 2 files, 219 lines
  - `easyasc/simulator/`: 7 files, 1586 lines
  - `easyasc/stub_functions/`: 8 files, 1023 lines
  - `easyasc/stub_functions/micro/`: 13 files, 1001 lines
  - `easyasc/stub_functions/vec/`: 14 files, 2306 lines
  - `easyasc/utils/`: 16 files, 3093 lines
  - `easyasc/resources/`: 2 Python helper files, 197 lines (plus non-Python resources)

## 2. End-to-End Flow (DSL to generated project)
1. `a2.py` / `a5.py` expose DSL APIs, data types, and stub entry points.
2. `decorators.py` wraps Python functions as `KernelBase` (`@kernel`) or `MicroModule` (`@vf`).
3. `pythonic.py` rewrites syntax sugar in AST form (name injection, boolean transforms, `if/elif/else` lowering).
4. Runtime objects (`Tensor`, `GMTensor`, `Var`, `Reg`, `MaskReg`) emit `Instruction` records.
5. `kernelbase/kernelbase.py` binds call arguments, captures outputs, and injects synchronization instructions.
6. `parser/asc.py` classifies instructions by side (cube/vec), prunes blocks, inserts auto-sync, and translates code.
7. `parser/asc_handlers/*.py` map each `Instruction` to Ascend C++ fragments.
8. `KernelBase.generate*` emits op host/kernel files, ACLNN test project artifacts, and helper shell scripts.
9. `torchplutin.OpExec` is the runtime entry helper:
   - Converts torch tensors/scalars into `GMTensor`/`Var`.
   - In simulator mode, clones each torch tensor into `GMTensor.data`.
   - Calls the kernel once to finalize bound metadata.
   - If `simulator=True`, invokes `KernelBase.run_sim(...)`.
   - In simulator mode, returns cloned torch outputs mapped from returned `GMTensor` views (`offset/span/step`) with output-shape restore.
   - Runs `KernelBase.generate(...)`.
   - Dumps input tensor `.bin` files and runs `b.sh` only when `simulator=False`.

## 3. Top-Level Files (`easyasc/`)
- `a2.py`: API aggregator for b3 profile.
- `a5.py`: extended API surface (Reg/MaskReg/CastConfig/micro ops), david profile.
- `decorators.py`: `kernel`, `func`, `auto_sync`, `vf`.
- `flowcontrol.py`: loop and conditional instruction emitters.
  - `unroll(...)` now delegates to Python builtin `range(...)` semantics (no DSL loop instruction emission).
- `globvars.py`: global runtime state (`active_kernel`, `active_micro`, tmp index, device settings).
- `pythonic.py`: AST transforms for DSL syntax sugar.
- `torchplutin.py`: `OpExec` execution helper and build/simulation orchestration.

## 4. Core Subsystems
### `kernelbase/`
- `kernelbase.py` (`KernelBase`):
  - Validates call arguments and records parameter metadata.
  - Emits initialization instructions (`create_gm_tensor`, `reset_cache`, helper vars/buffers).
  - Tracks output tensors from nested return structures.
  - Injects cross-core mutex synchronization instructions.
  - Provides `run_sim(...)` (simulator entry) and `generate(...)` (project/codegen path).
  - Supports `custom_op_path` in generation flows (`generate`, `generate_aclnn_test`, `generate_bashfiles`).

### `micro/`
- `micromodule.py` (`MicroModule`):
  - Restricts input types in micro context.
  - Clones call-time objects to avoid source metadata mutation.
  - Registers micro usage in kernel context (`call_micro`, `used_micros`).
  - Manages temporary reg/mask allocation and cast config defaults.

### `simulator/`
- `_core_utils.py`:
  - Provides shared non-negative integer validation for `core_idx`.
- `pipe.py` (`PipeBase`, `MTE2Pipe`, `MTE1Pipe`, `MPipe`, `FIXPipe`, `SimInstruction`):
  - Defines simulator cube pipe objects bound to a validated `core_idx`.
  - Provides per-pipe `SimInstruction` queues and enqueue/reset helpers.
  - Separates dispatch (`issue`) from execution (`execute_instruction`).
  - Executes `gm_to_l1_nd2nz` ND->NZ runtime data movement in `MTE2Pipe.execute_instruction(...)`.
  - Uses vectorized block copies for ND->NZ mapping (`reshape`/`permute`/`copy_`) to avoid Python per-element loops.
  - Executes `l1_to_l0` runtime movement in `MTE1Pipe.execute_instruction(...)` with dtype-aware NZ->NZ / NZ->ZN transforms.
  - `l1_to_l0` transpose path handles int8 with 32-row ZN tiles and other dtypes with 16-row tiles.
  - Executes `mmad` runtime movement in `MPipe.execute_instruction(...)` as NZ@NZ->NZ by decoding NZ to logical ND, computing `A @ B^T`, and encoding back to NZ.
  - `mmad` honors `is_init` accumulation semantics and uses destination-aware compute promotion for low-precision types.
  - Executes `l0c_to_gm_nz2nd` runtime movement in `FIXPipe.execute_instruction(...)` with NZ->ND decode, argument bounds validation, and GM writeback.
  - `l0c_to_gm_nz2nd` applies atomic add when atomic is enabled and emits a warning when `dst` dtype differs from the tracked atomic dtype.
  - Executes non-main-loop `sim_print` payloads in pipe phase with `[cube][core=<idx>][pipe=<name>]` log prefix.
  - `SimInstruction` carries dispatch sequence (`seq`) for deterministic cross-pipe scheduling.
- `cube.py` (`Cube`):
  - Holds cube-lane identity via validated `core_idx`.
  - Instantiates four cube pipes: `MTE2`, `MTE1`, `M`, `FIX`.
  - Receives `L1`, `L0A`, `L0B`, `L0C`, `UB1`, and `UB2` memory buffers from `Core`.
  - Uses `UB1`/`UB2` as shared references with `Vec0.UB`/`Vec1.UB`.
  - Executes cube instruction streams in `run(instructions)` with runtime loop/if interpretation.
  - Uses a two-phase simulator flow in `run(...)`: main pass for allocation/var/dispatch, then end-of-run pipe execution.
  - End-of-run pipe execution uses a sequence-aware scheduler and honors sync waits/sets (`waitflag`/`setflag`, `event_wait`/`event_set`/`event_release`) with deadlock detection.
  - Initializes event preset tokens from `create_sevent`/`create_devent`.
  - Tracks scalar results (`Var`) and local/gm tensor views in per-core dictionaries.
  - Treats `create_gm_tensor` as a direct GM data binding (`GMTensor.data`) without zero-fallback allocation.
  - Allocates tensor/DBuff views from local memory pools with per-position allocators and `reset_cache` reset.
  - Dispatches executable ops to `MTE2`/`MTE1`/`M`/`FIX` as `SimInstruction` records; pipe-side execution handles op-specific simulation.
  - `l1_to_l0` dispatch includes source transpose metadata (`src_is_transpose`) for pipe-side layout selection.
  - Handles `sim_print(pipe=Pipe.S)` in main loop while non-`S` prints are dispatched and executed during pipe phase.
  - Tracks atomic state in cube runtime and annotates only `l0c_to_gm_nz2nd` FIX instructions with current atomic status.
  - Supports simulator-only `sim_print` logging with `[cube][core=<idx>]` prefixes.
- `vec.py` (`Vec`):
  - Holds vector-lane identity via validated `core_idx`.
  - Receives `L1` (shared with `Cube.L1`) and `UB` memory buffers from `Core`.
  - Exposes `run(instructions)` placeholder for future vec-side simulation execution.
- `core.py` (`Core`):
  - Stores validated `core_idx`/`core_id`.
  - Allocates per-core simulator memories from `globvars` capacities in bytes (`cap * 1024`).
  - Builds one `Cube` instance and two `Vec` instances per `Core`, wiring shared `L1` and per-lane shared `UB` references.
  - Forwards simulator instructions and bound kernel arguments to its cube and vec units via `run(...)`.
- `base.py` (`SimulatorBase`):
  - Minimal simulator scaffold with `KernelBase`-typed constructor.
  - Stores kernel context and builds `cores` list from `device_type`.
  - Core-count mapping: `b3/b4 -> 20`, `b1/b2 -> 24`, `950 -> 32`.
  - `run()` inserts cube-side auto-sync instructions (`insert_auto_sync(..., mode='cube')`) before forwarding instructions and bound arguments into each core.

### `parser/`
- `asc.py`: instruction-side classification, pruning, and translation pipeline.
- `asc_autosync.py`: dependency-aware event insertion between producer/consumer pipelines.
- `asc_pruning.py`: block tree conversion and dead declaration/assignment elimination.
- `asc_utils.py`: dtype/position C++ mapping, expression folding, and offset expression builders.
- `helper.py`: code assembly helper utilities.

### `parser/asc_handlers/`
- Handler registry and per-op translators (`core`, `math_ops`, `events`, `flow`, `cube`, `vec_*`, `reinterpret`, etc.).
- `vec_micro_ops.py` maps micro instructions emitted in vector scope.
- `cube.py` keeps cube-op C++ lowering and now selects `l1_to_l0` transform op with `device_type`-aware branching (`b*` vs `950`) plus source transpose state.

### `stub_functions/`
- DSL instruction emission layer for:
  - scalar/variable ops (`var_op.py`)
  - tensor reinterpret/workspace split/simulator logging (`misc.py`)
  - cube operations (`cube.py`)
  - vector operations (`vec/*`)
  - micro register operations (`micro/*`)
  - synchronization/atomic/barrier primitives

### `utils/`
- Core semantic models and enums:
  - data types (`datatype.py`), positions (`positions.py`), pipes/events/mutex
  - IR node (`instruction.py`)
  - runtime values (`Var`, `Tensor`, `GMTensor`, `DBuff`)
  - `GMTensor` includes optional `data` for simulator-time torch value snapshots
  - register and expression systems (`Reg`, `RegList`, `MaskReg`, `RegOP`, `VecOP`)

## 5. Resource Templates
- `resources/CustomOp.tar.gz`: custom op project skeleton.
- `resources/CMakePresets.json`: CANN/toolkit preset template.
- `resources/tensorutils.h`, `resources/tensorx.h`, `resources/macros.h`: test/runtime helpers.
- `resources/setup_aclnn.py`, `resources/test.cpp`, `resources/parse_prof.py`: ACLNN test/profiling templates.

## 6. Current Notes
- `easyasc` follows a layered pattern: object model -> stub emitters -> parser handlers.
- Auto-sync insertion is active only inside explicit `start_auto_sync`/`end_auto_sync` regions.
- Generation helpers require at least one kernel invocation to collect bound argument/output metadata.
- Runtime/user-facing comments and messages in core touched files are standardized in English.
- For this repository snapshot, no standalone `README` file was found; root includes `AGENTS.md`, `ai_summary.md`, and a custom non-commercial `LICENSE`.
