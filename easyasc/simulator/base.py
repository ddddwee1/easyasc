from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from ..kernelbase.kernelbase import KernelBase
    from ..utils.instruction import Instruction
    from .core import Core


class SimulatorBase:
    def __init__(self, kernel: "KernelBase") -> None:
        from ..kernelbase.kernelbase import KernelBase
        from .. import globvars
        from .core import Core

        if not isinstance(kernel, KernelBase):
            raise TypeError(f"kernel must be KernelBase, got: {type(kernel)}")
        self.kernel = kernel
        self.device_type = str(getattr(globvars, "device_type", "")).lower()
        self.core_num = self._resolve_core_num(self.device_type)
        self.cores: List[Core] = [Core(core_idx) for core_idx in range(self.core_num)]

    @staticmethod
    def _resolve_core_num(device_type: str) -> int:
        if device_type in ("b3", "b4"):
            return 20
        if device_type in ("b1", "b2"):
            return 24
        if device_type == "950":
            return 32
        raise ValueError(f"Unsupported device_type for simulator: {device_type}")

    def run(self) -> None:
        from ..parser.asc_autosync import insert_auto_sync

        instructions: List["Instruction"] = self.kernel.instructions
        instructions = insert_auto_sync(list(instructions), mode="cube")
        bound_args = getattr(self.kernel, "_last_bound_args", None)
        for core in self.cores:
            core.run(instructions, bound_args=bound_args)
