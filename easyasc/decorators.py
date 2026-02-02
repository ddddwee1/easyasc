from typing import Callable, Any, Optional
from functools import wraps

from .kernelbase.kernelbase import KernelBase
from .pythonic import transform_kernel
from .utils.instruction import Instruction
from . import globvars


def _build_kernel(func: Callable[..., Any]) -> KernelBase:
    transformed = transform_kernel(func)
    return KernelBase(func.__name__, transformed)


class kernel:
    """Decorator class that returns a KernelBase wrapper."""
    def __init__(self, func: Optional[Callable[..., Any]] = None):
        self._kernel: Optional[KernelBase] = None
        if func is not None:
            inner = getattr(func, "__auto_sync_inner__", None)
            if inner is not None and callable(inner):
                transformed = transform_kernel(inner)
                sync = auto_sync()
                wrapped = sync._wrap(transformed)
                self._kernel = KernelBase(inner.__name__, wrapped)
            else:
                self._kernel = _build_kernel(func)

    def __call__(self, *args, **kwargs):
        if self._kernel is None:
            if len(args) == 1 and callable(args[0]) and not kwargs:
                target = args[0]
                inner = getattr(target, "__auto_sync_inner__", None)
                if inner is not None and callable(inner):
                    transformed = transform_kernel(inner)
                    sync = auto_sync()
                    wrapped = sync._wrap(transformed)
                    return KernelBase(inner.__name__, wrapped)
                return _build_kernel(target)
            raise TypeError("kernel decorator expects a single callable")
        return self._kernel(*args, **kwargs)

    def __getattr__(self, name):
        if self._kernel is not None:
            return getattr(self._kernel, name)
        raise AttributeError(name)


class func:
    """Decorator class that applies pythonic transform without creating a kernel."""
    def __init__(self, func: Optional[Callable[..., Any]] = None):
        self._func: Optional[Callable[..., Any]] = None
        if func is not None:
            inner = getattr(func, "__auto_sync_inner__", None)
            if inner is not None and callable(inner):
                transformed = transform_kernel(inner)
                sync = auto_sync()
                self._func = sync._wrap(transformed)
            else:
                self._func = transform_kernel(func)

    def __call__(self, *args, **kwargs):
        if self._func is None:
            if len(args) == 1 and callable(args[0]) and not kwargs:
                target = args[0]
                inner = getattr(target, "__auto_sync_inner__", None)
                if inner is not None and callable(inner):
                    transformed = transform_kernel(inner)
                    sync = auto_sync()
                    return sync._wrap(transformed)
                return transform_kernel(target)
            raise TypeError("func decorator expects a single callable")
        return self._func(*args, **kwargs)

    def __getattr__(self, name):
        if self._func is not None:
            return getattr(self._func, name)
        raise AttributeError(name)


class auto_sync:
    """Decorator/context that emits auto-sync markers around a block or call."""
    def __init__(self, func: Optional[Callable[..., Any]] = None):
        if func is not None and not callable(func):
            raise TypeError("auto_sync expects a callable")
        self._func: Optional[Callable[..., Any]] = func

    def _emit_start(self) -> None:
        if globvars.active_kernel is None:
            raise RuntimeError("auto_sync只能在kernel内使用")
        globvars.active_kernel.instructions.append(Instruction("start_auto_sync"))

    def _emit_end(self) -> None:
        if globvars.active_kernel is None:
            raise RuntimeError("auto_sync只能在kernel内使用")
        globvars.active_kernel.instructions.append(Instruction("end_auto_sync"))

    def _wrap(self, func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            self._emit_start()
            try:
                result = func(*args, **kwargs)
            except Exception:
                raise
            else:
                self._emit_end()
                return result
        wrapper.__auto_sync_inner__ = func  # type: ignore
        return wrapper

    def __call__(self, *args, **kwargs):
        if self._func is None:
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return self._wrap(args[0])
            raise TypeError("auto_sync decorator expects a single callable")
        self._emit_start()
        try:
            result = self._func(*args, **kwargs)
        except Exception:
            raise
        else:
            self._emit_end()
            return result

    def __enter__(self):
        self._emit_start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            return False
        self._emit_end()
        return True

    def __getattr__(self, name):
        if self._func is not None:
            return getattr(self._func, name)
        raise AttributeError(name)
