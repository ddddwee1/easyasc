## 2026-02-18 Step 1
- 阅读 `ai_summary.md` 并定位 `KernelBase.generate`、`generate_aclnn_test`、`generate_bashfiles` 的调用链与签名。
- 确认当前 `setup_aclnn.py` 与 `b.sh/r.sh` 生成逻辑里 custom op 路径均由 `cann_path` 间接推导。

## 2026-02-18 Step 2
- 在 `easyasc/kernelbase/kernelbase.py` 的 `generate` 新增参数 `custom_op_path: Optional[str] = None`，当为 `None` 时回退为 `cann_path`。
- 将 `custom_op_path` 传递到 `generate_aclnn_test` 和 `generate_bashfiles`。
- 在 `generate_aclnn_test` 与 `generate_bashfiles` 中新增同名入参与类型校验；默认逻辑保持与 `cann_path` 一致。
- 新增 `_resolve_custom_opp_path`，统一将路径规范为 `.../opp`（若已传入 `.../opp` 则不重复追加）。
- `generate_aclnn_test` 现在会同时改写 `setup_aclnn.py` 里的 `ascend_toolkit_install_path` 与 `custom_op_path`。
- `generate_bashfiles` 生成的 `b.sh/r.sh` 中 `LD_LIBRARY_PATH` 改为基于 `custom_op_path` 推导。

## 2026-02-18 Step 3
- 使用 `python -m py_compile easyasc/kernelbase/kernelbase.py` 完成语法校验，结果通过。

## 2026-02-18 Step 4
- 修复 `generate_aclnn_test` 中 `custom_op_path` 关键词匹配范围：仅替换赋值语句（`custom_op_path=` 或 `custom_op_path =`），避免误改 `setup_aclnn.py` 列表中的引用行。
- 重新执行 `python -m py_compile easyasc/kernelbase/kernelbase.py`，语法校验通过。

## 2026-02-18 Step 5
- 在 `easyasc/torchplutin.py` 的 `OpExec.__init__` 中新增参数 `custom_op_path: Optional[str] = None`，位置紧跟 `cann_path`。
- 新增实例字段 `self.custom_op_path`。
- 在 `OpExec.__call__` 中调用 `self.op_func.generate(...)` 时，新增透传 `custom_op_path=self.custom_op_path`，使 `OpExec` 可驱动新引入的 custom op 路径配置。
- 使用 `python -m py_compile easyasc/torchplutin.py` 完成语法校验，结果通过。
