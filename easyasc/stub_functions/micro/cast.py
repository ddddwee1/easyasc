from ...utils.reg import Reg, MaskReg
from ...utils.instruction import Instruction
from ...utils.castconfig import CastConfig
from .microutils import require_micro


def cast(dst: Reg, src: Reg, config: CastConfig, mask: MaskReg) -> None:
    micro = require_micro()
    if not isinstance(dst, Reg):
        raise TypeError(f"dst必须是Reg类型，当前类型: {type(dst)}")
    if not isinstance(src, Reg):
        raise TypeError(f"src必须是Reg类型，当前类型: {type(src)}")
    if not isinstance(config, CastConfig):
        raise TypeError(f"config必须是CastConfig类型，当前类型: {type(config)}")
    if not isinstance(mask, MaskReg):
        raise TypeError(f"mask必须是MaskReg类型，当前类型: {type(mask)}")
    if config.name == "":
        raise ValueError("cast config必须设置name")

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
