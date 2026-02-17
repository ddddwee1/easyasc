# easyasc 代码结构与文件详解（仅 `easyasc/`）

## 1. 范围与当前快照
- 本文只描述 `easyasc/` 目录下的目录和文件，不包含仓库其他目录。
- 当前 `easyasc/`（排除 `__pycache__`）共 100 个文件，其中 94 个 `.py` 文件，6 个资源文件（`.json`/`.h`/`.cpp`/`.tar.gz`）。
- Python 源码总行数约 14127 行。
- 目录级 Python 行数分布：
  - `easyasc/`: 7 文件，838 行
  - `easyasc/kernelbase/`: 1 文件，887 行
  - `easyasc/micro/`: 1 文件，180 行
  - `easyasc/parser/`: 5 文件，2083 行
  - `easyasc/parser/asc_handlers/`: 25 文件，2226 行
  - `easyasc/shortcuts/`: 2 文件，219 行
  - `easyasc/stub_functions/`: 8 文件，1032 行
  - `easyasc/stub_functions/micro/`: 13 文件，1036 行
  - `easyasc/stub_functions/vec/`: 14 文件，2335 行
  - `easyasc/utils/`: 16 文件，3094 行
  - `easyasc/resources/`: 2 个 Python 辅助脚本，197 行（另含 C/C++ 资源）
- 当前 `easyasc/` 根目录没有 `__init__.py`，存在 `a2.py` 和 `a5.py` 两套 API 汇总入口。

## 2. 目录清单（排除 `__pycache__`）
```text
easyasc/a2.py
easyasc/a5.py
easyasc/decorators.py
easyasc/flowcontrol.py
easyasc/globvars.py
easyasc/kernelbase/kernelbase.py
easyasc/micro/micromodule.py
easyasc/parser/asc.py
easyasc/parser/asc_autosync.py
easyasc/parser/asc_handlers/__init__.py
easyasc/parser/asc_handlers/atomic.py
easyasc/parser/asc_handlers/common.py
easyasc/parser/asc_handlers/core.py
easyasc/parser/asc_handlers/cube.py
easyasc/parser/asc_handlers/events.py
easyasc/parser/asc_handlers/flow.py
easyasc/parser/asc_handlers/math_ops.py
easyasc/parser/asc_handlers/misc.py
easyasc/parser/asc_handlers/pipe_ops.py
easyasc/parser/asc_handlers/reinterpret.py
easyasc/parser/asc_handlers/vec_binary.py
easyasc/parser/asc_handlers/vec_cast.py
easyasc/parser/asc_handlers/vec_compare.py
easyasc/parser/asc_handlers/vec_datamove.py
easyasc/parser/asc_handlers/vec_dupbrcb.py
easyasc/parser/asc_handlers/vec_gatherscatter.py
easyasc/parser/asc_handlers/vec_group.py
easyasc/parser/asc_handlers/vec_mask.py
easyasc/parser/asc_handlers/vec_micro.py
easyasc/parser/asc_handlers/vec_micro_ops.py
easyasc/parser/asc_handlers/vec_select.py
easyasc/parser/asc_handlers/vec_sort.py
easyasc/parser/asc_handlers/vec_unary.py
easyasc/parser/asc_handlers/vec_unary_scalar.py
easyasc/parser/asc_pruning.py
easyasc/parser/asc_utils.py
easyasc/parser/helper.py
easyasc/pythonic.py
easyasc/resources/CMakePresets.json
easyasc/resources/CustomOp.tar.gz
easyasc/resources/macros.h
easyasc/resources/parse_prof.py
easyasc/resources/setup_aclnn.py
easyasc/resources/tensorutils.h
easyasc/resources/tensorx.h
easyasc/resources/test.cpp
easyasc/shortcuts/__init__.py
easyasc/shortcuts/matmul.py
easyasc/stub_functions/__init__.py
easyasc/stub_functions/atomic.py
easyasc/stub_functions/barrier.py
easyasc/stub_functions/crosscore.py
easyasc/stub_functions/cube.py
easyasc/stub_functions/flags.py
easyasc/stub_functions/micro/__init__.py
easyasc/stub_functions/micro/arange.py
easyasc/stub_functions/micro/binary.py
easyasc/stub_functions/micro/cast.py
easyasc/stub_functions/micro/compare.py
easyasc/stub_functions/micro/datamove.py
easyasc/stub_functions/micro/dup.py
easyasc/stub_functions/micro/group.py
easyasc/stub_functions/micro/interleave.py
easyasc/stub_functions/micro/mask.py
easyasc/stub_functions/micro/microutils.py
easyasc/stub_functions/micro/unary.py
easyasc/stub_functions/micro/unaryscalar.py
easyasc/stub_functions/misc.py
easyasc/stub_functions/var_op.py
easyasc/stub_functions/vec/__init__.py
easyasc/stub_functions/vec/binary.py
easyasc/stub_functions/vec/cast.py
easyasc/stub_functions/vec/compare.py
easyasc/stub_functions/vec/datamove.py
easyasc/stub_functions/vec/dupbrcb.py
easyasc/stub_functions/vec/gatherscatter.py
easyasc/stub_functions/vec/group.py
easyasc/stub_functions/vec/select.py
easyasc/stub_functions/vec/sort.py
easyasc/stub_functions/vec/unary.py
easyasc/stub_functions/vec/unaryscalar.py
easyasc/stub_functions/vec/vecmask.py
easyasc/stub_functions/vec/vecutils.py
easyasc/torchplutin.py
easyasc/utils/Tensor.py
easyasc/utils/castconfig.py
easyasc/utils/comparemode.py
easyasc/utils/datatype.py
easyasc/utils/events.py
easyasc/utils/instruction.py
easyasc/utils/mask.py
easyasc/utils/mutex.py
easyasc/utils/pipe.py
easyasc/utils/positions.py
easyasc/utils/reg.py
easyasc/utils/regop.py
easyasc/utils/roundmode.py
easyasc/utils/selectmode.py
easyasc/utils/var.py
easyasc/utils/vecop.py
```

## 3. 核心执行链路（从 DSL 到生成代码）
1. `a2.py`/`a5.py` 汇总 DSL API、类型系统和 stub 函数入口。
2. `decorators.py` 提供 `@kernel`、`@func`、`@auto_sync`、`@vf`，将 Python 函数包装成 `KernelBase` 或 `MicroModule`。
3. `pythonic.py` 在 AST 层做语法糖改写：变量名自动注入、`and/or/not` 到位运算、`if/elif/else` 到 `with If/Elif/Else`。
4. 运行时对象（`Tensor`/`GMTensor`/`Var`/`Reg`/`MaskReg` 等）通过构造和 `<<=` 操作不断向 `active_kernel` 或 `active_micro` 追加 `Instruction`。
5. `kernelbase/kernelbase.py` 组织指令、调用 kernel、采集输出、插入 cross-core 同步、记录 workspace 形状。
6. `parser/asc.py` 将指令分为 cube/vec 两侧，做结构合法性检查、剪枝、自动同步插入和翻译。
7. `parser/asc_pruning.py`、`parser/asc_utils.py` 负责表达式折叠、临时变量内联、无用声明/变量/空块清理。
8. `parser/asc_handlers/*.py` 将每个 `Instruction` 变成 Ascend C++ 代码片段。
9. `KernelBase.generate*` 输出 `op_host`/`op_kernel`/ACLNN 测试脚手架，并依赖 `resources/` 模板资源生成工程。

## 4. 顶层文件说明（`easyasc/`）
- `a2.py`
  - API 聚合入口（Tensor/Var/事件/同步/vec/cube stub 函数等）。
  - 设置 `globvars.device_type = 'b3'`。
- `a5.py`
  - 相比 `a2.py` 额外暴露 `Reg`/`RegList`/`MaskReg`、`CastConfig`、`vf`、`stub_functions.micro` 全量接口。
  - 设置 `globvars.device_type = 'david'`。
- `decorators.py`
  - `_build_kernel()`：`transform_kernel()` 后构建 `KernelBase`。
  - `kernel`：装饰函数为可调用 `KernelBase`，兼容叠加 `auto_sync`。
  - `func`：只做 AST 变换，不建 kernel 对象。
  - `auto_sync`：支持 decorator 和 context manager，两端注入 `start_auto_sync` / `end_auto_sync`。
  - `vf`：将函数转为 `MicroModule`。
- `flowcontrol.py`
  - `range/unroll`：在 kernel 作用域发 `start_loop/end_loop`，在 micro 作用域发 `start_micro_loop/end_loop`。
  - `If/Elif/Else`：上下文管理器形式记录 `start_if/start_elif/start_else/end_if`。
- `globvars.py`
  - 运行态全局状态：`active_kernel`、`active_micro`、`tmp_idx`。
  - 原子态：`atomic_enabled`、`atomic_type`。
  - 设备与容量：`device_type`、`l1/l0a/l0b/l0c/ub` 容量。
- `pythonic.py`
  - `_VarNameAdder`：对 `Var/Tensor/DBuff/Reg/MaskReg/RegList/range/...` 自动注入 `name=`。
  - `_BoolOpRewriter`：`and/or/not` -> `&/|/~`。
  - `_IfRewriter`：把 `if/elif/else` 展开为 `with If/Elif/Else` 块。
  - `transform_kernel()`：保留原函数元数据并对齐源代码行号。
- `torchplutin.py`
  - `_run_bash_with_progress()`：执行 `b.sh` 并写 `b.sh.log`。
  - `OpExec`：把 `torch.Tensor` 和标量映射到 `GMTensor`/`Var`，调用 kernel，生成工程，导出输入 bin，可选自动执行构建。

## 5. `kernelbase/`
- `kernelbase/kernelbase.py`
  - `KernelBase.__call__`
    - 参数仅接受 `GMTensor` 或 `Var`。
    - 自动给参数对象回填参数名。
    - 为入参 `GMTensor` 插入 `create_gm_tensor`。
    - 自动创建 `_l0a/_l0b`（`DBuff`）与 `_l0acnt/_l0bcnt`（`Var`），供 matmul shortcut 复用。
    - 开头插入 `reset_cache`。
    - 执行函数后递归收集返回值中的输出 `GMTensor`。
    - 根据 `crosscore_mutex` 在头/尾补 `vec_ready/wait_vec/cube_ready/wait_cube`。
  - `dump_asc()`：输出 `{path}_cube.h`、`{path}_vec.h`。
  - `dump_kernel()`
    - 生成 cube/vec inline 函数封装。
    - 自动注入 `TPipe`、`pipe_ptr`、`_offset`。
    - 自动 include `used_micros` 对应头文件。
    - 生成 `.cpp` 入口，完成 tiling 读取与 AIC/AIV 分发。
  - `generate_op_host()`
    - 生成 `{name}_tiling.h`。
    - 生成 op 定义 `.cpp`，注册输入输出/属性、tiling 函数、workspace 大小计算。
    - `workspace_shapes` 会折叠到 `userWorkspaceSize`。
  - `generate_op_project()`
    - 解压 `resources/CustomOp.tar.gz`。
    - 重写 `CMakePresets.json` 的 `ASCEND_CANN_PACKAGE_PATH`。
    - 按 `device_type` 设置 `ASCEND_COMPUTE_UNIT`（`ascend910b` 或 `ascend910_93`）。
  - `generate()`
    - 统一产物流程：`op_host` + `op_kernel` + `tensorutils.h` 复制 + ACLNN 测试工程 + `b.sh/r.sh`。
  - `generate_aclnn_test()`
    - 复制 `macros.h`、`parse_prof.py`，改写 `setup_aclnn.py` 的 toolkit 路径和架构标签。
    - 改写 `tensorx.h` include 到 `aclnn_{name}.h`。
    - 基于最后一次绑定参数自动写 `test.cpp`。
    - `profile=True` 时会从 `resources/test.cpp` 提取并启用 profiling 代码段。
  - `generate_bashfiles()`
    - 生成构建脚本 `b.sh` 和运行脚本 `r.sh`。

## 6. `micro/`
- `micro/micromodule.py`
  - `TempRegStatus`：管理可复用临时寄存器占用状态。
  - `MicroModule`
    - `__call__` 仅允许 `Tensor`(UB) 和 `Var` 入参。
    - 调用时克隆入参对象，防止污染原对象元数据。
    - 在 kernel 中调用 micro 时会记录 `call_micro` 指令并登记到 `active_kernel.used_micros`。
    - 提供 `get_reg/release_reg` 临时寄存器池。
    - 提供 `get_mask` 默认掩码寄存器复用。
    - `get_default_cast_cfg` 维护默认 `CastConfig`。
    - `gen_code()` 输出 micro 头文件，包含 `CastTrait` 常量、`__VEC_SCOPE__` 包装和翻译后的指令代码。

## 7. `parser/`
- `parser/asc.py`
  - 指令侧别分类：根据 stub 来源、handler 模块、pipe/event 信息判定 cube/vec/both。
  - `split_instructions()`：分侧后做空块剪枝、无用声明剪枝、无用变量剪枝。
  - `translate()`：执行折叠策略（循环折叠、声明+赋值折叠、create_var 与 event 调序）并逐条调用 handler。
  - `translate_split()`：先分侧，再插入 auto-sync，再做内存占用分析并翻译。
  - `analyze_usage()`：按 reset-cache 分段统计 `create_tensor/create_dbuf` 的空间占用并打印表格。
- `parser/asc_autosync.py`
  - `AutosyncNode` 递归解析 loop/if 结构，识别跨 pipe 的 producer/consumer。
  - 根据是否单缓冲决定使用 `SEvent` 或 `DEvent`。
  - 自动插入 `event_wait/event_set` 并批量生成临时 event 声明。
  - `insert_auto_sync()` 只在 `start_auto_sync` 到 `end_auto_sync` 包围区间内生效。
- `parser/asc_pruning.py`
  - 把线性指令解析成块树（loop/if/inst）。
  - 删除不产出代码的空块。
  - 删除无用 `create_*` 声明（含事件和 GM/Tensor 声明）。
  - 按 side（cube/vec）做变量依赖追踪和反向保活，清除无用 Var/赋值/切片临时定义。
- `parser/asc_utils.py`
  - dtype/position 到 C++ 映射。
  - `value_to_cpp()` 支持 `Expr`、`Var`、`Tensor`、`GMTensor`、`DBuff`。
  - `build_expr_state()` 收集并内联 `_tmp_var/_tmp_tensor/_tmp_gmtensor`。
  - `build_offset_expr` 和 `build_offset_expr_nz` 负责切片线性偏移表达式生成。
  - 可选依赖 `sympy` 做表达式简化。
- `parser/helper.py`
  - `CodeHelper`：简洁缩进与字符串拼接器。

## 8. `parser/asc_handlers/` 文件职责
- `__init__.py`：集中注册全部 handler；并把 `vec_micro_ops.py` 的 `MICRO_OP_HANDLERS` 合并进总映射。
- `common.py`：统一类型导入、工具函数导入，定义 `Handler` 类型；`start_auto_sync/end_auto_sync` handler 当前为空实现。
- `core.py`：`create_*`、`get_buf`、`slice_tensor`、`slice_gm_tensor`、`micro_slice_tensor`、`split_workspace` 转 C++。
- `flow.py`：`start_loop/start_micro_loop/end_loop/start_if/start_elif/start_else/end_if`。
- `math_ops.py`：block index/num、对齐、CeilDiv/Min/Max、标量 sqrt、四则运算。
- `cube.py`：`GM2L1_ND2NZ`、`L0NZ2*`、`MMAD`、`L0C2GM_NZ2ND`、`L0C2L1`。
- `events.py`：`SEvent/DEvent` 声明与 `set/wait/setall/release`。
- `pipe_ops.py`：`READY/WAIT/ALL*`、`PipeBarrier`、`SetFlag/WaitFlag`。
- `atomic.py`：原子类型与开启关闭。
- `reinterpret.py`：`ReinterpretCast`。
- `misc.py`：`reset_cache` -> `pipe_ptr->Reset(); OccupyMMTE1Events();`。
- `vec_binary.py`：add/sub/mul/div/max/min/and/or/muladddst。
- `vec_unary.py`：exp/ln/abs/rec/sqrt/rsqrt/vnot/relu。
- `vec_unary_scalar.py`：adds/muls/vmaxs/vmins/lrelu/axpy。
- `vec_dupbrcb.py`：dup/brcb。
- `vec_gatherscatter.py`：gather/scatter。
- `vec_group.py`：cadd/cmax/cmin/cg*/cpadd。
- `vec_sort.py`：sort32/mergesort4/mergesort_2seq。
- `vec_mask.py`：set_mask/reset_mask。
- `vec_cast.py`：cast。
- `vec_compare.py`：compare/compare_scalar/set_cmpmask。
- `vec_select.py`：select（支持 tensor/tensor 与 tensor/scalar 两种模式）。
- `vec_datamove.py`：gm_to_ub_pad/ub_to_gm_pad/ub_to_ub。
- `vec_micro.py`：`call_micro` 参数桥接。
- `vec_micro_ops.py`：所有 `micro_*` 指令的 C++ 发射器（算术/掩码/搬运/cast/compare/select/gather/scatter/interleave 等）。

## 9. `shortcuts/`
- `shortcuts/__init__.py`：仅导出 `matmul`。
- `shortcuts/matmul.py`
  - 基于 `KernelBase` 内置 `_l0a/_l0b/_l0acnt/_l0bcnt` 的高层矩阵乘法快捷函数。
  - 支持 `splitn` 和 `splitk` 两种拆分路径。
  - 兼容 L1 转置视图访问。
  - 在某些整数路径可通过 `reinterpret` 做 int4 适配。
  - `is_init` 支持常量布尔和动态 Var 条件。

## 10. `stub_functions/`（DSL 指令发射层）
- `stub_functions/__init__.py`：聚合 var/cube/vec/micro/同步/原子/屏障接口。
- `stub_functions/var_op.py`
  - `CeilDiv/GetCubeNum/GetCubeIdx/GetVecNum/GetVecIdx/GetSubBlockIdx`。
  - `Align16/32/64/128/256`、`scalar_sqrt`。
  - `var_mul/var_add/var_sub/var_div/Min/Max`，带基础常量值传播。
  - 在 `active_micro` 优先向 micro 指令流写入（否则写 kernel）。
- `stub_functions/cube.py`
  - `gm_to_l1_nd2nz/l1_to_l0/mmad/l0c_to_gm_nz2nd/l0c_to_l1`。
  - 支持从 slice 信息推断 M/N/K/stride 参数。
  - 与原子态联动：必要时自动发 `set_atomic_type`。
- `stub_functions/misc.py`
  - `reinterpret`：更新 shape/span（按 C0 比例换算）并发 reinterpret 指令。
  - `split_workspace`：按 shape 计算 numel，登记 `workspace_shapes` 并返回 `GMTensor` 视图。
  - `reset_cache`：发 reset_cache 指令。
- `stub_functions/atomic.py`
  - `atomic_add/atomic_max/atomic_min` 上下文管理器，维护 `globvars.atomic_enabled/atomic_type`。
- `stub_functions/barrier.py`
  - `bar_m/bar_v/bar_mte3/bar_mte2/bar_mte1/bar_fix/bar_all`。
- `stub_functions/flags.py`
  - `setflag/waitflag`（带 src/dst pipe 和 event_id）。
- `stub_functions/crosscore.py`
  - `cube_ready/vec_ready/wait_cube/wait_vec/all*` 一组 cross-core 原语。

## 11. `stub_functions/vec/`
- `vec/__init__.py`：聚合 vec 运算 API。
- `vec/vecutils.py`：repeat 推断、stride 推断与校验工具。
- `vec/binary.py`：add/sub/mul/div/vmax/vmin/vand/vor/muladddst。
- `vec/unary.py`：exp/ln/abs/rec/sqrt/rsqrt/vnot/relu。
- `vec/unaryscalar.py`：adds/muls/vmaxs/vmins/lrelu/axpy。
- `vec/dupbrcb.py`：`dup`、`brcb`。
- `vec/gatherscatter.py`：`gather`、`scatter`。
- `vec/group.py`：`cmax/cmin/cadd/cgmax/cgmin/cgadd/cpadd`，包含一次性使用提醒。
- `vec/sort.py`：`sort32`、`mergesort4`、`mergesort_2seq`。
- `vec/vecmask.py`：`set_mask/reset_mask`。
- `vec/datamove.py`：`gm_to_ub_pad/ub_to_gm_pad/ub_to_ub`，支持自动参数推导。
- `vec/cast.py`：vec cast 的 repeat/stride/round_mode 推导。
- `vec/compare.py`：tensor-tensor 和 tensor-scalar compare，另有 `set_cmpmask`。
- `vec/select.py`：按 `SelectMode` 发射 select 指令，支持 tensor/scalar 源。

## 12. `stub_functions/micro/`
- `micro/__init__.py`：聚合全部 micro API。
- `micro/microutils.py`：`require_micro`、`ensure_mask`、`format_scalar`、`dtype_size`。
- `micro/arange.py`：生成索引序列（增序/减序）。
- `micro/binary.py`：二元寄存器算子（vmax/vmin/add/sub/mul/div/vand/vor/vxor/prelu）。
- `micro/unary.py`：一元寄存器算子（exp/abs/relu/sqrt/ln/log/log2/log10/neg/vnot/vcopy）。
- `micro/unaryscalar.py`：寄存器 + 标量（vmaxs/vmins/adds/muls/lrelu/shiftls/shiftrs/axpy）。
- `micro/group.py`：c*/cg*/cpadd。
- `micro/dup.py`：标量/寄存器复制到寄存器。
- `micro/cast.py`：寄存器 cast，支持 `CastConfig`。
- `micro/compare.py`：compare（reg/reg 或 reg/scalar）与 select。
- `micro/mask.py`：mask 位运算、搬运、pack/unpack、interleave、update、move_to_spr。
- `micro/interleave.py`：寄存器 interleave/deinterleave。
- `micro/datamove.py`
  - 定义 `LoadDist`/`StoreDist`。
  - `ub_to_reg/reg_to_ub`。
  - 连续分布搬运：downsample/upsample/unpack/unpack4/brcb/single/pack4。
  - `ub_to_reg_gather/reg_to_ub_scatter/gather/gather_mask`。

## 13. `utils/`（核心数据模型与语义）
- `utils/instruction.py`：`Instruction(opname, **kwargs)` 指令对象。
- `utils/datatype.py`：`DataTypeValue` 与 `Datatype` 常量（含 `int64/uint*`）。
- `utils/positions.py`：`Position` 与 C++ 映射。
- `utils/pipe.py`：`Pipe`（MTE2/MTE1/M/V/FIX/MTE3/S/ALL）。
- `utils/mask.py`：`MaskType` 枚举。
- `utils/roundmode.py`：`RoundMode`。
- `utils/comparemode.py`：`CompareMode`。
- `utils/selectmode.py`：`SelectMode`。
- `utils/events.py`
  - `SEvent/DEvent`，支持 `set/wait/setall/release`。
  - 构造时可自动向 active_kernel 注入 `create_sevent/create_devent`。
- `utils/mutex.py`
  - `CvMutex/VcMutex`，可在构造时自动登记到 active_kernel 的 `crosscore_mutex`。
  - 支持 `lock/ready/wait/free` 映射到 crosscore 原语。
- `utils/var.py`
  - `Var` 自动推断 dtype，支持算术和比较。
  - `Expr` 用于条件表达式，禁止直接参与 Python 布尔判断。
- `utils/Tensor.py`
  - `Tensor`
    - 2D 局部张量，构造会发 `create_tensor`。
    - `<<=` 自动选择 gm/ub/cube/micro datamove 或 vec/reg/regop 路径。
    - 支持 `VecOP` 语法糖、`downsample/upsample/unpack/unpack4/brcb/single` 读取模式标记。
    - `__getitem__` 支持切片并记录 `slice_tensor` 或 `micro_slice_tensor`。
  - `DBuff`
    - 双缓冲抽象，`dbuf[idx]` 返回 `Tensor` 视图并记录 `get_buf`。
  - `GMTensor`
    - 全局张量抽象，支持切片、`<<=` 写回、互斥绑定与同步方法。
- `utils/vecop.py`
  - `VecOP` 延迟记录 vec 表达式，在 `dst <<= ...` 时落地到 stub 函数。
  - `maximum/minimum` 语法糖。
- `utils/castconfig.py`
  - `CastConfig`（round_mode/reg_layout/saturate/name）。
  - 在 micro 中创建时自动挂入 `cast_cfg_list`。
- `utils/reg.py`
  - `Reg`：寄存器对象，支持算术、比较、cast、group、gather、`<<=` 赋值语义。
  - `RegList`：寄存器数组，支持向量化 `RegOP` 批量发射和规约入口。
  - `MaskReg`：掩码寄存器，支持位操作、选择、pack/unpack、更新。
- `utils/regop.py`
  - `RegOP`：可组合寄存器表达式对象。
  - `emit()` 覆盖 micro 算术/掩码/比较/选择/搬运/gather/scatter 全路径。
  - `run_regop()` 可在 micro 内物化到临时寄存器并自动复用/释放。

## 14. `resources/`
- `resources/CustomOp.tar.gz`
  - CANN 自定义算子项目模板压缩包。
- `resources/CMakePresets.json`
  - 模板 CMake preset，含 `ASCEND_CANN_PACKAGE_PATH`、`ASCEND_COMPUTE_UNIT`。
- `resources/tensorutils.h`
  - 生成 `op_kernel` 时复制到输出目录供内核代码使用。
- `resources/macros.h`
  - ACLNN 测试宏（`CHECK_RET/PREPARE_OP/EXECOP`）和 dtype 宏定义。
- `resources/tensorx.h`
  - `TensorX<T, aclDataType>` 模板，封装 host/device 内存、拷贝和 `aclTensor` 构造。
- `resources/setup_aclnn.py`
  - ACLNN testcase 的本地编译脚本模板（会被 `KernelBase.generate_aclnn_test` 改写路径与架构）。
- `resources/parse_prof.py`
  - msprof 结果解析脚本，提取目标 OP 耗时并导出过滤后的 trace。
- `resources/test.cpp`
  - profiling 代码片段模板源（`generate_aclnn_test(profile=True)` 会抽取并反注释指定块）。

## 15. 当前代码组织特征与注意点
- `easyasc/` 采用“对象方法 + stub API + parser handler”三层结构，指令语义集中在 `Instruction`。
- `auto_sync` 机制依赖显式区间标记；不在 `start_auto_sync/end_auto_sync` 内不会做自动事件插入。
- `Var/Tensor/GMTensor` 的临时对象命名和表达式折叠是代码简化关键，核心在 `asc_utils.build_expr_state()`。
- `KernelBase.generate*` 依赖“先执行过一次 kernel”来拿到绑定参数和输出集合。
- micro 路径强调类型与位置约束：Tensor 必须 UB，Reg/MaskReg 只能在 `active_micro` 存在时工作。
- `a2.py` 与 `a5.py` 是两套不同设备/API聚合入口；`a5.py` 包含更完整的 micro/reg 能力。
