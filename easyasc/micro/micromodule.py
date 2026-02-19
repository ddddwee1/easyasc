import inspect
from typing import Any, Callable, Dict, List, Tuple

from .. import globvars
from ..utils.datatype import DataTypeValue
from ..utils.instruction import Instruction
from ..utils.mask import MaskType
from ..utils.reg import MaskReg, Reg
from ..utils.Tensor import Tensor
from ..utils.var import Var
from ..utils.positions import Position
from ..utils.castconfig import CastConfig


class TempRegStatus:
    def __init__(self, idx: int, dtype: DataTypeValue):
        self.reg = object.__new__(Reg)
        self.reg.name = f'_tmp_reg_{idx}'
        self.reg.dtype = dtype
        self.valid = True 
    
    def lock(self):
        self.valid = False 
    
    def release(self):
        self.valid = True 


class MicroModule:
    def __init__(self, name: str, func: Callable[..., Any]) -> None:
        self.func = func
        self.name = name
        self.instructions: List[Any] = []
        self.input_list: List[Any] = []
        self.tmp_idx = 0
        self.tmp_masks: Dict[str, MaskReg] = {}
        self.default_cast_cfg = None 
        self.cast_cfg_list: List[Any] = []
        self.tmp_regs: Dict[str, List[TempRegStatus]] = {}

    def get_default_cast_cfg(self):
        if self.default_cast_cfg is not None:
            return self.default_cast_cfg
        self.default_cast_cfg = CastConfig(name='default_castcfg')
        self.cast_cfg_list
        return self.default_cast_cfg

    def get_reg(self, dtype: DataTypeValue):
        if dtype.name not in self.tmp_regs:
            self.tmp_regs[dtype.name] = []
        for i in self.tmp_regs[dtype.name]:
            if i.valid:
                i.lock()
                return i.reg 
        new_stat = TempRegStatus(self.tmp_idx, dtype)
        self.tmp_idx += 1 
        self.tmp_regs[dtype.name].append(new_stat)
        new_stat.lock()
        return new_stat.reg

    def release_reg(self, reg: Reg):
        for i in self.tmp_regs[reg.dtype.name]:
            if i.reg.name == reg.name:
                i.release()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs:
            raise TypeError("MicroModule does not support keyword arguments")
        for arg in args:
            if not isinstance(arg, (Tensor, Var)):
                raise TypeError(f"MicroModule only accepts Tensor or Var inputs, got: {type(arg)}")
            if isinstance(arg, Tensor) and arg.position is not Position.UB:
                raise ValueError(f"MicroModule only accepts Tensor inputs in UB, current position: {arg.position}")
        sig = inspect.signature(self.func)
        try:
            bound = sig.bind(*args)
        except TypeError as exc:
            raise TypeError(f"MicroModule call argument mismatch: {exc}") from exc
        micro_args: List[Any] = []
        for name, arg in bound.arguments.items():
            if isinstance(arg, Tensor):
                cloned = object.__new__(Tensor)
                cloned.__dict__ = {
                    k: (v.copy() if isinstance(v, list) else v)
                    for k, v in arg.__dict__.items()
                }
                cloned.name = name
                micro_args.append(cloned)
            elif isinstance(arg, Var):
                cloned = object.__new__(Var)
                cloned.__dict__ = arg.__dict__.copy()
                cloned.name = name
                micro_args.append(cloned)
            else:
                raise TypeError(f"MicroModule only accepts Tensor or Var inputs, got: {type(arg)}")
        if not self.input_list:
            self.input_list = micro_args
        if globvars.active_kernel is not None:
            globvars.active_kernel.used_micros.add(self)
            globvars.active_kernel.instructions.append(
                Instruction("call_micro", name=self.name, args=list(args))
            )
        globvars.active_micro = self
        self.func(*micro_args)
        for _,v in self.tmp_masks.items():
            self.instructions.insert(0, Instruction("create_maskreg", reg=v))
        globvars.active_micro = None

    def get_mask(self, dtype: DataTypeValue) -> MaskReg:
        if not isinstance(dtype, DataTypeValue):
            raise TypeError(f"dtype must be DataTypeValue, got: {type(dtype)}")
        key = dtype.name
        if key not in self.tmp_masks:
            mask = object.__new__(MaskReg)
            mask.dtype = dtype
            mask.init_mode = MaskType.ALL
            idx = self.tmp_idx
            self.tmp_idx += 1
            mask.name = f"_tmp_maskreg_{idx}"
            self.tmp_masks[key] = mask
        return self.tmp_masks[key]

    def gen_code(self, path: str) -> None:
        if not isinstance(path, str):
            raise TypeError(f"path must be str, got: {type(path)}")
        from ..parser.asc import translate
        from ..parser.asc_utils import dtype_to_cpp
        from ..parser.helper import CodeHelper
        from ..utils.castconfig import CastConfig

        helper = CodeHelper()
        helper("#pragma once")
        helper('#include "tensorutils.h"')
        helper()

        for cfg in self.cast_cfg_list:
            if not isinstance(cfg, CastConfig):
                continue
            sat = "SAT" if cfg.saturate else "NO_SAT"
            helper(
                "static constexpr MicroAPI::CastTrait "
                f"{cfg.name} = {{ MicroAPI::RegLayout::{cfg.reg_layout}, "
                f"MicroAPI::SatMode::{sat}, MicroAPI::MaskMergeMode::ZEROING, "
                f"RoundMode::{cfg.round_mode} }};"
            )
        if self.cast_cfg_list:
            helper()

        args = []
        for arg in self.input_list:
            if isinstance(arg, Tensor):
                args.append(
                    f"__ubuf__ {dtype_to_cpp(arg.dtype)}* {arg.name}"
                )
            elif isinstance(arg, Var):
                args.append(
                    f"const {dtype_to_cpp(arg.dtype)} &{arg.name}"
                )
            else:
                raise TypeError(f"MicroModule only accepts Tensor or Var inputs, got: {type(arg)}")
        args_str = ", ".join(args)

        helper(f"__aicore__ inline void {self.name}({args_str}){{")
        helper.ir()
        helper("__VEC_SCOPE__")
        helper("{")
        helper.ir()

        code = translate(self.instructions)
        for line in code.splitlines():
            helper(line)

        helper.il()
        helper("}")
        helper.il()
        helper("}")
        helper()

        with open(path, "w", encoding="utf-8") as f:
            f.write(str(helper))
