from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .kernelbase.kernelbase import KernelBase
    from .micro.micromodule import MicroModule
    from .utils.datatype import DataTypeValue

active_kernel: Optional["KernelBase"] = None
active_micro: Optional["MicroModule"] = None
tmp_idx: int = 0
atomic_enabled: bool = False
atomic_type: Optional["DataTypeValue"] = None
device_type: str = "b3"

l1_cap = 512
l0a_cap = 64 
l0b_cap = 64 
l0c_cap = 128
ub_cap = 192
