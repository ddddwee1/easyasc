# easyasc Architecture Summary (`easyasc/` only)

## 1. Scope and Snapshot
- This document summarizes only the `easyasc/` directory.
- Current snapshot (excluding `__pycache__`):
  - Total files: 101
  - Python files: 94
  - Python source lines: 14086
- Python line distribution by directory:
  - `easyasc/`: 7 files, 845 lines
  - `easyasc/kernelbase/`: 1 file, 930 lines
  - `easyasc/micro/`: 1 file, 180 lines
  - `easyasc/parser/`: 5 files, 2083 lines
  - `easyasc/parser/asc_handlers/`: 25 files, 2223 lines
  - `easyasc/shortcuts/`: 2 files, 219 lines
  - `easyasc/stub_functions/`: 8 files, 1012 lines
  - `easyasc/stub_functions/micro/`: 13 files, 1001 lines
  - `easyasc/stub_functions/vec/`: 14 files, 2306 lines
  - `easyasc/utils/`: 16 files, 3090 lines
  - `easyasc/resources/`: 2 Python helper files, 197 lines (plus non-Python resources)

## 2. End-to-End Flow (DSL to generated project)
1. `a2.py` / `a5.py` expose DSL APIs, types, and stub function entry points.
2. `decorators.py` wraps Python functions as `KernelBase` (`@kernel`) or `MicroModule` (`@vf`).
3. `pythonic.py` rewrites AST syntax sugar (name injection, boolean op rewriting, `if/elif/else` lowering).
4. Runtime objects (`Tensor`, `GMTensor`, `Var`, `Reg`, `MaskReg`) emit `Instruction` records.
5. `kernelbase/kernelbase.py` manages instructions, captures outputs, and injects synchronization.
6. `parser/asc.py` splits instructions by side (cube/vec), prunes blocks, inserts auto-sync, and translates.
7. `parser/asc_handlers/*.py` map each `Instruction` to Ascend C++ fragments.
8. `KernelBase.generate*` emits op host/kernel files, ACLNN test project, and shell scripts.

## 3. Top-Level Files (`easyasc/`)
- `a2.py`: API aggregator for b3 profile.
- `a5.py`: extended API surface (Reg/MaskReg/CastConfig/micro ops), david profile.
- `decorators.py`: `kernel`, `func`, `auto_sync`, `vf`.
- `flowcontrol.py`: loop and conditional instruction emitters.
- `globvars.py`: global runtime state (`active_kernel`, `active_micro`, tmp index, device settings).
- `pythonic.py`: AST transformations.
- `torchplutin.py`: `OpExec` execution helper and build orchestration.

## 4. Core Subsystems
### `kernelbase/`
- `kernelbase.py` (`KernelBase`):
  - Validates call arguments and records parameter metadata.
  - Emits initialization instructions (`create_gm_tensor`, `reset_cache`, helper vars/buffers).
  - Tracks output tensors from nested return structures.
  - Handles cross-core mutex synchronization insertion.
  - Generates host/kernel/test project artifacts.

### `micro/`
- `micromodule.py` (`MicroModule`):
  - Restricts argument types for micro context.
  - Clones call-time objects to avoid source metadata mutation.
  - Registers micro usage in kernel context (`call_micro`, `used_micros`).
  - Manages temporary reg/mask allocation and cast config defaults.

### `parser/`
- `asc.py`: instruction-side classification, pruning, translation pipeline.
- `asc_autosync.py`: dependency-aware event insertion between producer/consumer pipelines.
- `asc_pruning.py`: block tree conversion and dead declaration/assignment elimination.
- `asc_utils.py`: dtype/position C++ mapping, expression folding, offset expression builders.
- `helper.py`: code assembly helper utilities.

### `parser/asc_handlers/`
- Handler registry and per-op translators (`core`, `math_ops`, `events`, `flow`, `cube`, `vec_*`, `reinterpret`, etc.).
- `vec_micro_ops.py` provides mapping for micro instructions emitted in vector scope.

### `stub_functions/`
- DSL instruction emission layer for:
  - scalar/variable ops (`var_op.py`)
  - tensor reinterpret/workspace split (`misc.py`)
  - cube operations (`cube.py`)
  - vector operations (`vec/*`)
  - micro register operations (`micro/*`)
  - synchronization/atomic/barrier primitives

### `utils/`
- Core semantic models and enums:
  - data types (`datatype.py`), positions (`positions.py`), pipes/events/mutex
  - IR node (`instruction.py`)
  - runtime values (`Var`, `Tensor`, `GMTensor`, `DBuff`)
  - register and expression systems (`Reg`, `RegList`, `MaskReg`, `RegOP`, `VecOP`)

## 5. Resource Templates
- `resources/CustomOp.tar.gz`: custom op project skeleton.
- `resources/CMakePresets.json`: CANN/toolkit preset template.
- `resources/tensorutils.h`, `resources/tensorx.h`, `resources/macros.h`: test/runtime helpers.
- `resources/setup_aclnn.py`, `resources/test.cpp`, `resources/parse_prof.py`: ACLNN test/profiling templates.

## 6. Integrated Recent Changes (from `ai_modifications.md`)
### 2026-02-18
- Added `custom_op_path` support through the generation stack:
  - `KernelBase.generate(..., custom_op_path=None)` now accepts an explicit custom op install path.
  - `custom_op_path` is propagated into `generate_aclnn_test` and `generate_bashfiles`.
  - `_resolve_custom_opp_path` normalizes incoming paths to `.../opp`.
  - `setup_aclnn.py` rewriting now updates both toolkit path and custom op path assignment.
  - generated `b.sh` / `r.sh` now derive `LD_LIBRARY_PATH` from `custom_op_path`.
- Added `custom_op_path` passthrough in `torchplutin.OpExec` (`__init__` and `__call__`).
- Syntax checks for touched files were completed with `py_compile`.

### 2026-02-19
- Converted Chinese runtime messages/docstrings to English in core modules:
  - `easyasc/decorators.py`
  - `easyasc/flowcontrol.py`
  - `easyasc/micro/micromodule.py`
  - `easyasc/kernelbase/kernelbase.py`
  - `easyasc/torchplutin.py`
- Repaired syntax/indentation regressions in stub modules after bulk text replacement:
  - `easyasc/stub_functions/var_op.py`
  - `easyasc/stub_functions/misc.py`
  - `easyasc/stub_functions/vec/select.py`
  - `easyasc/stub_functions/vec/vecutils.py`
  - `easyasc/stub_functions/micro/datamove.py`
- Completed full syntax validation of all `easyasc` Python modules:
  - `python -m py_compile $(rg --files easyasc -g '*.py')`

## 7. Current Notes
- `easyasc` follows a layered pattern: object model -> stub emitters -> parser handlers.
- Auto-sync insertion is active only inside explicit `start_auto_sync`/`end_auto_sync` regions.
- Generation helpers require at least one kernel invocation to collect bound argument/output metadata.
- For this repository snapshot, no standalone `README` file was found; markdown files in root are `AGENTS.md` and `ai_summary.md`.
