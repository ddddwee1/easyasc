from typing import Optional

from ...utils.datatype import Datatype
from ...utils.reg import MaskReg
from ...utils.var import Var
from ...utils.instruction import Instruction
from .microutils import ensure_mask, require_micro


def mask_not(dst: MaskReg, src: MaskReg, mask: Optional[MaskReg] = None) -> None:
    micro = require_micro()
    if not isinstance(dst, MaskReg):
        raise TypeError(f"dst must be MaskReg type, current type: {type(dst)}")
    if not isinstance(src, MaskReg):
        raise TypeError(f"src must be MaskReg type, current type: {type(src)}")
    if dst.dtype != src.dtype:
        raise ValueError("dst/src data types must match") 
    mask = ensure_mask(mask, dst.dtype, micro)
    micro.instructions.append(Instruction("micro_masknot", dst=dst, src=src, mask=mask))


def mask_and(dst: MaskReg, src1: MaskReg, src2: MaskReg, mask: Optional[MaskReg] = None) -> None:
    micro = require_micro()
    for name, reg in ("dst", dst), ("src1", src1), ("src2", src2):
        if not isinstance(reg, MaskReg):
            raise TypeError(f"{name} must be MaskReg type, current type: {type(reg)}")
        if dst.dtype != src1.dtype or dst.dtype != src2.dtype:
            raise ValueError("dst/src data types must match") 
    mask = ensure_mask(mask, dst.dtype, micro)
    micro.instructions.append(
        Instruction("micro_maskand", dst=dst, src1=src1, src2=src2, mask=mask)
    )


def mask_or(dst: MaskReg, src1: MaskReg, src2: MaskReg, mask: Optional[MaskReg] = None) -> None:
    micro = require_micro()
    for name, reg in ("dst", dst), ("src1", src1), ("src2", src2):
        if not isinstance(reg, MaskReg):
            raise TypeError(f"{name} must be MaskReg type, current type: {type(reg)}")
        if dst.dtype != src1.dtype or dst.dtype != src2.dtype:
            raise ValueError("dst/src data types must match") 
    mask = ensure_mask(mask, dst.dtype, micro)
    micro.instructions.append(
        Instruction("micro_maskor", dst=dst, src1=src1, src2=src2, mask=mask)
    )


def mask_xor(dst: MaskReg, src1: MaskReg, src2: MaskReg, mask: Optional[MaskReg] = None) -> None:
    micro = require_micro()
    for name, reg in ("dst", dst), ("src1", src1), ("src2", src2):
        if not isinstance(reg, MaskReg):
            raise TypeError(f"{name} must be MaskReg type, current type: {type(reg)}")
        if dst.dtype != src1.dtype or dst.dtype != src2.dtype:
            raise ValueError("dst/src data types must match") 
    mask = ensure_mask(mask, dst.dtype, micro)
    micro.instructions.append(
        Instruction("micro_maskxor", dst=dst, src1=src1, src2=src2, mask=mask)
    )


def mask_mov(dst: MaskReg, src: MaskReg, mask: Optional[MaskReg] = None) -> None:
    micro = require_micro()
    if not isinstance(dst, MaskReg):
        raise TypeError(f"dst must be MaskReg type, current type: {type(dst)}")
    if not isinstance(src, MaskReg):
        raise TypeError(f"src must be MaskReg type, current type: {type(src)}")
    if dst.dtype != src.dtype:
        raise ValueError("dst/src data types must match") 
    mask = ensure_mask(mask, dst.dtype, micro)
    micro.instructions.append(Instruction("micro_maskmov", dst=dst, src=src, mask=mask))


def mask_interleave(dst0: MaskReg, dst1: MaskReg, src0: MaskReg, src1: MaskReg) -> None:
    micro = require_micro()
    for name, reg in ("dst0", dst0), ("dst1", dst1), ("src0", src0), ("src1", src1):
        if not isinstance(reg, MaskReg):
            raise TypeError(f"{name} must be MaskReg type, current type: {type(reg)}")
        if dst0.dtype != dst1.dtype or src0.dtype != src1.dtype or dst0.dtype != src0.dtype:
            raise ValueError("dst/src data types must match") 
    micro.instructions.append(
        Instruction(
            "micro_maskinterl", dst0=dst0, dst1=dst1, src0=src0, src1=src1
        )
    )


def mask_deinterleave(dst0: MaskReg, dst1: MaskReg, src0: MaskReg, src1: MaskReg) -> None:
    micro = require_micro()
    for name, reg in ("dst0", dst0), ("dst1", dst1), ("src0", src0), ("src1", src1):
        if not isinstance(reg, MaskReg):
            raise TypeError(f"{name} must be MaskReg type, current type: {type(reg)}")
        if dst0.dtype != dst1.dtype or src0.dtype != src1.dtype or dst0.dtype != src0.dtype:
            raise ValueError("dst/src data types must match") 
    micro.instructions.append(
        Instruction(
            "micro_maskdeinterl", dst0=dst0, dst1=dst1, src0=src0, src1=src1
        )
    )


def mask_sel(dst: MaskReg, src1: MaskReg, src2: MaskReg, mask: Optional[MaskReg] = None) -> None:
    micro = require_micro()
    for name, reg in ("dst", dst), ("src1", src1), ("src2", src2):
        if not isinstance(reg, MaskReg):
            raise TypeError(f"{name} must be MaskReg type, current type: {type(reg)}")
        if dst.dtype != src1.dtype or dst.dtype != src2.dtype:
            raise ValueError("dst/src data types must match") 
    mask = ensure_mask(mask, dst.dtype, micro)
    micro.instructions.append(
        Instruction("micro_masksel", dst=dst, src1=src1, src2=src2, mask=mask)
    )


def mask_pack(dst: MaskReg, src: MaskReg, low_part: bool = True) -> None:
    micro = require_micro()
    if not isinstance(dst, MaskReg):
        raise TypeError(f"dst must be MaskReg type, current type: {type(dst)}")
    if not isinstance(src, MaskReg):
        raise TypeError(f"src must be MaskReg type, current type: {type(src)}")
    if dst.dtype != src.dtype:
        raise ValueError("dst/src data types must match") 
    mode = "MicroAPI::HighLowPart::LOWEST" if low_part else "MicroAPI::HighLowPart::HIGHEST"
    micro.instructions.append(Instruction("micro_maskpack", dst=dst, src=src, mode=mode))


def mask_unpack(dst: MaskReg, src: MaskReg, low_part: bool = True) -> None:
    micro = require_micro()
    if not isinstance(dst, MaskReg):
        raise TypeError(f"dst must be MaskReg type, current type: {type(dst)}")
    if not isinstance(src, MaskReg):
        raise TypeError(f"src must be MaskReg type, current type: {type(src)}")
    if dst.dtype != src.dtype:
        raise ValueError("dst/src data types must match") 
    mode = "MicroAPI::HighLowPart::LOWEST" if low_part else "MicroAPI::HighLowPart::HIGHEST"
    micro.instructions.append(Instruction("micro_maskunpack", dst=dst, src=src, mode=mode))


def move_mask_spr(dst: MaskReg) -> None:
    micro = require_micro()
    if not isinstance(dst, MaskReg):
        raise TypeError(f"dst must be MaskReg type, current type: {type(dst)}") 
    micro.instructions.append(Instruction("micro_movemaskspr", dst=dst))


def update_mask(dst: MaskReg, cnt: Var) -> None:
    micro = require_micro()
    if not isinstance(dst, MaskReg):
        raise TypeError(f"dst must be MaskReg type, current type: {type(dst)}")
    if not isinstance(cnt, Var):
        raise TypeError(f"cnt must be Var type, current type: {type(cnt)}")
    if cnt.dtype is not Datatype.uint32:
        raise ValueError("update_mask only supports uint32counter") 
    micro.instructions.append(Instruction("micro_updatemask", dst=dst, cnt=cnt))
