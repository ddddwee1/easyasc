from typing import Optional

from ..utils.instruction import Instruction
from ..utils.var import Var
from ..utils.datatype import Datatype, DataTypeValue
from .. import globvars


class atomic_add:
    def __init__(self, cond: Optional[Var] = None, dtype: DataTypeValue = Datatype.float):
        if cond is not None:
            raise NotImplementedError("当前框架不支持带条件的atomic_add")
        if not isinstance(dtype, DataTypeValue):
            raise TypeError(f"dtype必须是DataTypeValue类型，当前类型: {type(dtype)}")
        if dtype is not Datatype.float:
            raise NotImplementedError("当前框架仅支持float原子类型")
        self.dtype = dtype

    def __enter__(self):
        if globvars.active_kernel is None:
            raise RuntimeError("atomic_add只能在kernel内使用")
        globvars.active_kernel.instructions.append(Instruction("atomic_add"))
        globvars.atomic_enabled = True
        globvars.atomic_type = self.dtype

    def __exit__(self, exec_type, exec_val, exec_traceback):
        if exec_type:
            return False
        if globvars.active_kernel is None:
            raise RuntimeError("atomic_add只能在kernel内使用")
        globvars.active_kernel.instructions.append(Instruction("atomic_end"))
        globvars.atomic_enabled = False
        globvars.atomic_type = None
        return True


class atomic_max:
    def __init__(self, dtype: DataTypeValue = Datatype.float):
        if not isinstance(dtype, DataTypeValue):
            raise TypeError(f"dtype必须是DataTypeValue类型，当前类型: {type(dtype)}")
        if dtype is not Datatype.float:
            raise NotImplementedError("当前框架仅支持float原子类型")
        self.dtype = dtype

    def __enter__(self):
        if globvars.active_kernel is None:
            raise RuntimeError("atomic_max只能在kernel内使用")
        globvars.active_kernel.instructions.append(Instruction("atomic_max"))
        globvars.atomic_enabled = True
        globvars.atomic_type = self.dtype

    def __exit__(self, exec_type, exec_val, exec_traceback):
        if exec_type:
            return False
        if globvars.active_kernel is None:
            raise RuntimeError("atomic_max只能在kernel内使用")
        globvars.active_kernel.instructions.append(Instruction("atomic_end"))
        globvars.atomic_enabled = False
        globvars.atomic_type = None
        return True


class atomic_min:
    def __init__(self, dtype: DataTypeValue = Datatype.float):
        if not isinstance(dtype, DataTypeValue):
            raise TypeError(f"dtype必须是DataTypeValue类型，当前类型: {type(dtype)}")
        if dtype is not Datatype.float:
            raise NotImplementedError("当前框架仅支持float原子类型")
        self.dtype = dtype

    def __enter__(self):
        if globvars.active_kernel is None:
            raise RuntimeError("atomic_min只能在kernel内使用")
        globvars.active_kernel.instructions.append(Instruction("atomic_min"))
        globvars.atomic_enabled = True
        globvars.atomic_type = self.dtype

    def __exit__(self, exec_type, exec_val, exec_traceback):
        if exec_type:
            return False
        if globvars.active_kernel is None:
            raise RuntimeError("atomic_min只能在kernel内使用")
        globvars.active_kernel.instructions.append(Instruction("atomic_end"))
        globvars.atomic_enabled = False
        globvars.atomic_type = None
        return True
