from typing import Callable, Any

from .kernelbase.kernelbase import KernelBase
from .pythonic import transform_kernel



def kernel(func: Callable[..., Any]) -> KernelBase:
    """Decorator that returns a KernelBase wrapper."""
    transformed = transform_kernel(func)
    kernel_obj = KernelBase(func.__name__, transformed)
    return kernel_obj


def func(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that applies pythonic transform without creating a kernel."""
    return transform_kernel(func)
