from ...utils.reg import Reg, MaskReg
from ...utils.instruction import Instruction
from ...utils.castconfig import CastConfig
from .microutils import require_micro, ensure_mask
from typing import Optional


def cast(dst: Reg, src: Reg, config: Optional[CastConfig] = None, mask: Optional[MaskReg] = None) -> None:
    micro = require_micro()
    mask = ensure_mask(mask, dst.dtype, micro)
    if config is None:
        config = micro.get_default_cast_cfg()
    if not isinstance(dst, Reg):
        raise TypeError(f"dst must be Reg type, current type: {type(dst)}")
    if not isinstance(src, Reg):
        raise TypeError(f"src must be Reg type, current type: {type(src)}")
    if not isinstance(config, CastConfig):
        raise TypeError(f"config must be CastConfig type, current type: {type(config)}")
    if not isinstance(mask, MaskReg):
        raise TypeError(f"mask must be MaskReg type, current type: {type(mask)}")
    if config.name == "":
        raise ValueError("cast config must set name")

    micro.instructions.append(
        Instruction(
            "micro_cast",
            dst=dst,
            src=src,
            config=config,
            mask=mask,
            ddst=dst.dtype,
            dsrc=src.dtype,
        )
    )
