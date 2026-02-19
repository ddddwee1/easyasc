from typing import Optional, Union

from ...utils.datatype import Datatype as DT
from ...utils.reg import Reg, MaskReg
from ...utils.Tensor import Tensor
from ...utils.var import Var
from ...utils.positions import Position
from ...utils.instruction import Instruction
from .microutils import dtype_size, ensure_mask, require_micro


class LoadDistValue:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"LoadDistValue({self.name!r})"

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LoadDistValue) and self.name == other.name


class LoadDist:
    DOWNSAMPLE = LoadDistValue("DOWNSAMPLE")
    UPSAMPLE = LoadDistValue("UPSAMPLE")
    SINGLE_VALUE = LoadDistValue("SINGLE_VALUE")
    BRCB = LoadDistValue("BRCB")
    UNPACK = LoadDistValue("UNPACK")
    UNPACK4 = LoadDistValue("UNPACK4")


class StoreDistValue:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"StoreDistValue({self.name!r})"

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, StoreDistValue) and self.name == other.name


class StoreDist:
    DOWNSAMPLE = StoreDistValue("DOWNSAMPLE")
    PACK4 = StoreDistValue("PACK4")
    SINGLE_VALUE = StoreDistValue("SINGLE_VALUE")
    NORMAL = StoreDistValue("NORMAL")


# micro register related data move
def ub_to_reg(
    dst: Reg,
    src: Tensor,
    blk_stride: Union[int, Var] = 1,
    mask: Optional[MaskReg] = None,
) -> None:
    micro = require_micro()
    if not isinstance(src, Tensor):
        raise TypeError(f"src must be Tensor type, current type: {type(src)}")
    if src.position is not Position.UB:
        raise ValueError(f"src must be at UB position, current position: {src.position}")
    if not isinstance(dst, Reg):
        raise TypeError(f"dst must be Reg type, current type: {type(dst)}")
    if dst.dtype != src.dtype:
        raise ValueError("dst/src data types must match") 
    mask = ensure_mask(mask, dst.dtype, micro)
    micro.instructions.append(
        Instruction("micro_ub2reg", dst=dst, src=src, blk_stride=blk_stride, mask=mask)
    )


def reg_to_ub(
    dst: Tensor,
    src: Reg,
    blk_stride: Union[int, Var] = 1,
    mask: Optional[MaskReg] = None,
) -> None:
    micro = require_micro()
    if not isinstance(dst, Tensor):
        raise TypeError(f"dst must be Tensor type, current type: {type(dst)}")
    if dst.position is not Position.UB:
        raise ValueError(f"dst must be at UB position, current position: {dst.position}")
    if not isinstance(src, Reg):
        raise TypeError(f"src must be Reg type, current type: {type(src)}")
    if dst.dtype != src.dtype:
        raise ValueError("dst/src data types must match") 
    mask = ensure_mask(mask, dst.dtype, micro)
    micro.instructions.append(
        Instruction("micro_reg2ub", dst=dst, src=src, blk_stride=blk_stride, mask=mask)
    )


def ub_to_reg_continuous(dst: Reg, src: Tensor, loaddist: LoadDistValue) -> None:
    micro = require_micro()
    if not isinstance(src, Tensor):
        raise TypeError(f"src must be Tensor type, current type: {type(src)}")
    if src.position is not Position.UB:
        raise ValueError(f"src must be at UB position, current position: {src.position}")
    if not isinstance(dst, Reg):
        raise TypeError(f"dst must be Reg type, current type: {type(dst)}")
    if not isinstance(loaddist, LoadDistValue):
        raise TypeError(f"loaddist must be LoadDistValue type, current type: {type(loaddist)}")
    if dst.dtype != src.dtype:
        raise ValueError("dst/src data types must match") 
    size = dtype_size(dst.dtype)
    if loaddist == LoadDist.DOWNSAMPLE:
        if size == 1:
            mode = "DIST_DS_B8"
        elif size == 2:
            mode = "DIST_DS_B16"
        else:
            raise ValueError("downsample does not support 32-bit type")
    elif loaddist == LoadDist.UPSAMPLE:
        if size == 1:
            mode = "DIST_US_B8"
        elif size == 2:
            mode = "DIST_US_B16"
        else:
            raise ValueError("upsample does not support 32-bit type")
    elif loaddist == LoadDist.SINGLE_VALUE:
        if size == 1:
            mode = "DIST_BRC_B8"
        elif size == 2:
            mode = "DIST_BRC_B16"
        elif size == 4:
            mode = "DIST_BRC_B32"
        else:
            raise ValueError(f"{dst.dtype} does not support single_value")
    elif loaddist == LoadDist.BRCB:
        if size == 2:
            mode = "DIST_E2B_B16"
        elif size == 4:
            mode = "DIST_E2B_B32"
        else:
            raise ValueError(f"{dst.dtype} does not support brcb")
    elif loaddist == LoadDist.UNPACK:
        if size == 1:
            mode = "DIST_UNPACK_B8"
        elif size == 2:
            mode = "DIST_UNPACK_B16"
        elif size == 4:
            mode = "DIST_UNPACK_B32"
        else:
            raise ValueError(f"{dst.dtype} does not support unpack")
    elif loaddist == LoadDist.UNPACK4:
        if size == 1:
            mode = "DIST_UNPACK4_B8"
        else:
            raise ValueError("unpack4 only supports 8-bit type")
    else:
        raise ValueError(f"unknown loaddist: {loaddist}")
    micro.instructions.append(
        Instruction("micro_ub2regcont", dst=dst, src=src, mode=mode)
    )


def reg_to_ub_continuous(
    dst: Tensor,
    src: Reg,
    mask: Optional[MaskReg],
    storedist: StoreDistValue,
) -> None:
    micro = require_micro()
    if not isinstance(dst, Tensor):
        raise TypeError(f"dst must be Tensor type, current type: {type(dst)}")
    if dst.position is not Position.UB:
        raise ValueError(f"dst must be at UB position, current position: {dst.position}")
    if not isinstance(src, Reg):
        raise TypeError(f"src must be Reg type, current type: {type(src)}")
    if not isinstance(storedist, StoreDistValue):
        raise TypeError(f"storedist must be StoreDistValue type, current type: {type(storedist)}")
    if dst.dtype != src.dtype:
        raise ValueError("dst/src data types must match") 
    mask = ensure_mask(mask, dst.dtype, micro)

    size = dtype_size(dst.dtype)
    if storedist == StoreDist.DOWNSAMPLE:
        if size == 1:
            mode = "DIST_PACK_B16"
        elif size == 2:
            mode = "DIST_PACK_B32"
        elif size == 4:
            mode = "DIST_PACK_B64"
        else:
            raise ValueError("downsample does not support 64-bit type")
    elif storedist == StoreDist.PACK4:
        if size == 1:
            mode = "DIST_PACK4_B32"
        else:
            raise ValueError("pack4 only supports 8-bit type")
    elif storedist == StoreDist.SINGLE_VALUE:
        if size == 1:
            mode = "DIST_FIRST_ELEMENT_B8"
        elif size == 2:
            mode = "DIST_FIRST_ELEMENT_B16"
        elif size == 4:
            mode = "DIST_FIRST_ELEMENT_B32"
        else:
            raise ValueError(f"{dst.dtype} does not support single_value")
    elif storedist == StoreDist.NORMAL:
        if size == 1:
            mode = "DIST_NORM_B8"
        elif size == 2:
            mode = "DIST_NORM_B16"
        elif size == 4:
            mode = "DIST_NORM_B32"
        else:
            raise ValueError(f"{dst.dtype} does not support normal")
    else:
        raise ValueError(f"unknown storedist: {storedist}")
    micro.instructions.append(
        Instruction("micro_reg2ubcont", dst=dst, src=src, mask=mask, mode=mode)
    )


def reg_to_ub_downsample(dst: Tensor, src: Reg, mask: Optional[MaskReg] = None) -> None:
    reg_to_ub_continuous(dst, src, mask, StoreDist.DOWNSAMPLE)


def reg_to_ub_pack4(dst: Tensor, src: Reg, mask: Optional[MaskReg] = None) -> None:
    reg_to_ub_continuous(dst, src, mask, StoreDist.PACK4)


def reg_to_ub_single(dst: Tensor, src: Reg, mask: Optional[MaskReg] = None) -> None:
    reg_to_ub_continuous(dst, src, mask, StoreDist.SINGLE_VALUE)


def ub_to_reg_single(dst: Reg, src: Tensor) -> None:
    ub_to_reg_continuous(dst, src, LoadDist.SINGLE_VALUE)


def ub_to_reg_upsample(dst: Reg, src: Tensor) -> None:
    ub_to_reg_continuous(dst, src, LoadDist.UPSAMPLE)


def ub_to_reg_downsample(dst: Reg, src: Tensor) -> None:
    ub_to_reg_continuous(dst, src, LoadDist.DOWNSAMPLE)


def ub_to_reg_unpack(dst: Reg, src: Tensor) -> None:
    ub_to_reg_continuous(dst, src, LoadDist.UNPACK)


def ub_to_reg_unpack4(dst: Reg, src: Tensor) -> None:
    ub_to_reg_continuous(dst, src, LoadDist.UNPACK4)


def ub_to_reg_brcb(dst: Reg, src: Tensor) -> None:
    ub_to_reg_continuous(dst, src, LoadDist.BRCB)


# Gather and scatter
def ub_to_reg_gather(
    dst: Reg,
    src: Tensor,
    index: Reg,
    mask: Optional[MaskReg] = None,
) -> None:
    micro = require_micro()
    if not isinstance(dst, Reg):
        raise TypeError(f"dst must be Reg type, current type: {type(dst)}")
    if not isinstance(src, Tensor):
        raise TypeError(f"src must be Tensor type, current type: {type(src)}")
    if not isinstance(index, Reg):
        raise TypeError(f"index must be Reg type, current type: {type(index)}")
    if dst.dtype != src.dtype:
        raise ValueError("dst/src data types must match") 
    mask = ensure_mask(mask, src.dtype, micro)
    size = dtype_size(src.dtype)
    if size == 2:
        if index.dtype != DT.uint16:
            raise ValueError("index must be uint16")
    elif size == 4:
        if index.dtype != DT.uint32:
            raise ValueError("index must be uint32")
    micro.instructions.append(
        Instruction("micro_datacopygather", dst=dst, src=src, index=index, mask=mask)
    )


def reg_to_ub_scatter(
    dst: Tensor,
    src: Reg,
    index: Reg,
    mask: Optional[MaskReg] = None,
) -> None:
    micro = require_micro()
    if not isinstance(dst, Tensor):
        raise TypeError(f"dst must be Tensor type, current type: {type(dst)}")
    if not isinstance(src, Reg):
        raise TypeError(f"src must be Reg type, current type: {type(src)}")
    if not isinstance(index, Reg):
        raise TypeError(f"index must be Reg type, current type: {type(index)}")
    if dst.dtype != src.dtype:
        raise ValueError("dst/src data types must match") 
    mask = ensure_mask(mask, src.dtype, micro)
    size = dtype_size(src.dtype)
    if size == 2:
        if index.dtype != DT.uint16:
            raise ValueError("index must be uint16")
    elif size == 4:
        if index.dtype != DT.uint32:
            raise ValueError("index must be uint32")
    micro.instructions.append(
        Instruction("micro_datacopyscatter", dst=dst, src=src, index=index, mask=mask)
    )


def gather(dst: Reg, src: Reg, index: Reg) -> None:
    micro = require_micro()
    if not isinstance(dst, Reg):
        raise TypeError(f"dst must be Reg type, current type: {type(dst)}")
    if not isinstance(src, Reg):
        raise TypeError(f"src must be Reg type, current type: {type(src)}")
    if not isinstance(index, Reg):
        raise TypeError(f"index must be Reg type, current type: {type(index)}")
    if dst.dtype != src.dtype:
        raise ValueError("dst/src data types must match") 
    size = dtype_size(src.dtype)
    if size == 2:
        if index.dtype != DT.uint16:
            raise ValueError("index must be uint16")
    elif size == 4:
        if index.dtype != DT.uint32:
            raise ValueError("index must be uint32")
    micro.instructions.append(
        Instruction("micro_gather", dst=dst, src=src, index=index)
    )


def gather_mask(dst: Reg, src: Reg, mask: Optional[MaskReg] = None) -> None:
    micro = require_micro()
    if not isinstance(dst, Reg):
        raise TypeError(f"dst must be Reg type, current type: {type(dst)}")
    if not isinstance(src, Reg):
        raise TypeError(f"src must be Reg type, current type: {type(src)}")
    if dst.dtype != src.dtype:
        raise ValueError("dst/src data types must match") 
    mask = ensure_mask(mask, src.dtype, micro)
    micro.instructions.append(
        Instruction("micro_gathermask", dst=dst, src=src, mask=mask)
    )
