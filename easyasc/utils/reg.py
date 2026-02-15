from typing import Optional

from .. import globvars
from .datatype import DataTypeValue
from .instruction import Instruction
from .mask import MaskType, MaskTypeValue


class Reg:
    def __init__(self, dtype: DataTypeValue, name: str = "") -> None:
        if not isinstance(dtype, DataTypeValue):
            raise TypeError(f"dtype必须是DataTypeValue类型，当前类型: {type(dtype)}")
        if not isinstance(name, str):
            raise TypeError(f"name必须是str类型，当前类型: {type(name)}")

        if name == "":
            module = globvars.active_micro
            if module is None:
                raise RuntimeError("active_micro为None，无法自动生成Reg名称")
            idx = module.tmp_idx
            module.tmp_idx += 1
            name = f"_reg_{idx}"

        self.dtype = dtype
        self.name = name
        if globvars.active_micro is not None:
            globvars.active_micro.instructions.append(Instruction("create_reg", reg=self))

    def __repr__(self) -> str:
        return f"Reg(name={self.name!r}, dtype={self.dtype!r})"

    def __str__(self) -> str:
        return self.name


class MaskReg:
    def __init__(
        self,
        dtype: DataTypeValue,
        init_mode: Optional[MaskTypeValue] = None,
        name: str = "",
    ) -> None:
        if not isinstance(dtype, DataTypeValue):
            raise TypeError(f"dtype必须是DataTypeValue类型，当前类型: {type(dtype)}")
        if init_mode is None:
            init_mode = MaskType.ALL
        if not isinstance(init_mode, MaskTypeValue):
            raise TypeError(f"init_mode必须是MaskTypeValue类型，当前类型: {type(init_mode)}")
        if not isinstance(name, str):
            raise TypeError(f"name必须是str类型，当前类型: {type(name)}")

        if name == "":
            module = globvars.active_micro
            if module is None:
                raise RuntimeError("active_micro为None，无法自动生成MaskReg名称")
            idx = module.tmp_idx
            module.tmp_idx += 1
            name = f"_maskreg_{idx}"

        self.dtype = dtype
        self.init_mode = init_mode
        self.name = name
        if globvars.active_micro is not None:
            globvars.active_micro.instructions.append(Instruction("create_maskreg", reg=self))

    def __repr__(self) -> str:
        return (
            f"MaskReg(name={self.name!r}, dtype={self.dtype!r}, "
            f"init_mode={self.init_mode!r})"
        )

    def __str__(self) -> str:
        return self.name
