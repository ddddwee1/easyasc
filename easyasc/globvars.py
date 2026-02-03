from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .kernelbase.kernelbase import KernelBase
    from .utils.datatype import DataTypeValue

active_kernel: Optional["KernelBase"] = None
tmp_idx: int = 0
atomic_enabled: bool = False
atomic_type: Optional["DataTypeValue"] = None
device_type: str = "b3"
