from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union

if TYPE_CHECKING:
    from .kernelbase.kernelbase import KernelBase


def _run_bash_with_progress(script_path: str, log_path: str) -> None:
    import os
    import subprocess
    import sys
    import time

    if not os.path.exists(script_path) or not os.path.isfile(script_path):
        raise FileNotFoundError(f"{script_path} does not exist or is not a file, cannot execute")

    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            ["bash", script_path],
            stdout=log_file,
            stderr=log_file,
        )
        start_ts = time.time()
        bar_width = 24
        while True:
            ret = proc.poll()
            elapsed = time.time() - start_ts
            pos = int(elapsed * 8) % (2 * bar_width)
            if pos >= bar_width:
                pos = 2 * bar_width - pos - 1
            bar = [" "] * bar_width
            bar[pos] = "#"
            msg = f"\r[{''.join(bar)}] bash b.sh running... {elapsed:6.1f}s"
            print(msg, end="", file=sys.stderr, flush=True)
            if ret is not None:
                break
            time.sleep(0.2)
        print("\r" + " " * (bar_width + 40) + "\r", end="", file=sys.stderr, flush=True)
        if ret != 0:
            raise RuntimeError(
                f"bash {script_path} failed with return code {ret}. Log: {log_path}"
            )


class OpExec:
    def __init__(
        self,
        op_func: Union["KernelBase", Callable[..., Any]],
        out_dir: str = "",
        cann_path: Optional[str] = None,
        custom_op_path: Optional[str] = None,
        profile: bool = False,
        gen_only: bool = False,
    ) -> None:
        self.op_func = op_func
        self.out_dir = out_dir
        self.cann_path = cann_path
        self.custom_op_path = custom_op_path
        self.profile = profile
        self.gen_only = gen_only 

    def __call__(self, *args: Any) -> None:
        from .kernelbase.kernelbase import KernelBase
        from .utils.Tensor import GMTensor
        from .utils.datatype import Datatype
        from .utils.var import Var
        import os

        if not isinstance(self.op_func, KernelBase):
            raise TypeError("op_func must be a KernelBase instance")

        try:
            import torch
        except Exception as exc:
            raise ImportError("OpExec.__call__ requires torch") from exc

        tensor_args: List[Any] = []
        scalar_vars: List[Var] = []
        seen_scalar = False
        for idx, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                if seen_scalar:
                    raise TypeError("Invalid argument order: torch.Tensor must come before int/float")
                tensor_args.append(arg)
                continue
            if isinstance(arg, (int, float)):
                seen_scalar = True
                scalar_vars.append(Var(arg))
                continue
            raise TypeError(f"Invalid argument type: arg #{idx} has type {type(arg)}")

        dtype_map = {
            torch.float16: Datatype.half,
            torch.float32: Datatype.float,
            torch.int32: Datatype.int,
            torch.int64: Datatype.int64,
            torch.int8: Datatype.int8,
            torch.uint8: Datatype.uint8,
            torch.int16: Datatype.int16,
            torch.bfloat16: Datatype.bfloat16,
        }
        if hasattr(torch, "uint16"):
            dtype_map[torch.uint16] = Datatype.uint16  # type: ignore
        if hasattr(torch, "uint32"):
            dtype_map[torch.uint32] = Datatype.uint32  # type: ignore
        if hasattr(torch, "uint64"):
            dtype_map[torch.uint64] = Datatype.uint64  # type: ignore

        gm_tensors: List[GMTensor] = []
        for tensor in tensor_args:
            inferred_shape: List[Any] = []
            for dim in tensor.shape:
                matched = None
                for scalar_var in scalar_vars:
                    if scalar_var.value == dim:
                        matched = scalar_var
                        break
                if matched is None:
                    inferred_shape.append(int(dim))
                else:
                    inferred_shape.append(matched)

            dtype = dtype_map.get(tensor.dtype)
            if dtype is None:
                raise TypeError(f"Unsupported torch dtype: {tensor.dtype}")
            gm_tensors.append(GMTensor(dtype, inferred_shape))

        self.op_func(*(gm_tensors + scalar_vars))
        self.op_func.generate(
            self.out_dir,
            cann_path=self.cann_path,
            custom_op_path=self.custom_op_path,
            profile=self.profile,
        )

        if not self.op_func._last_bound_args:
            raise ValueError("op_func has not been executed; cannot extract IO info from KernelBase")

        output_gmtensors = self.op_func._last_output_gmtensors
        param_names = list(self.op_func._last_bound_args.keys())
        gmtensor_param_names: List[str] = []
        for name in param_names:
            bound_val = self.op_func._last_bound_args.get(name)
            if isinstance(bound_val, GMTensor):
                gmtensor_param_names.append(name)

        input_param_names = []
        for name in gmtensor_param_names:
            bound_val = self.op_func._last_bound_args.get(name)
            if bound_val not in output_gmtensors:
                input_param_names.append(name)

        if len(tensor_args) == len(gmtensor_param_names):
            tensor_by_name = {
                name: tensor_args[idx] for idx, name in enumerate(gmtensor_param_names)
            }
        elif len(tensor_args) == len(input_param_names):
            tensor_by_name = {
                name: tensor_args[idx] for idx, name in enumerate(input_param_names)
            }
        else:
            raise ValueError(
                "Number of torch.Tensor inputs does not match KernelBase GMTensor parameters"
            )

        base_dir = self.out_dir if self.out_dir else self.op_func.name
        input_dir = f"{base_dir}_aclnn_test/input"
        os.makedirs(input_dir, exist_ok=True)

        for name in input_param_names:
            tensor = tensor_by_name[name]
            tensor_cpu = tensor.detach().cpu()
            tensor_bytes = tensor_cpu.contiguous().view(torch.uint8)
            tensor_bytes.numpy().tofile(os.path.join(input_dir, f"input_{name}.bin"))

        if not self.gen_only:
            log_path = os.path.abspath("b.sh.log")
            _run_bash_with_progress("b.sh", log_path)
