from ...utils.castconfig import CastConfig
from .arange import arange
from .binary import vmax, vmin, add, sub, mul, div, vand, vor, vxor, prelu
from .cast import cast
from .compare import compare, select
from .datamove import (
    LoadDist,
    LoadDistValue,
    StoreDist,
    StoreDistValue,
    ub_to_reg,
    reg_to_ub,
    ub_to_reg_continuous,
    reg_to_ub_continuous,
    reg_to_ub_downsample,
    reg_to_ub_pack4,
    reg_to_ub_single,
    ub_to_reg_single,
    ub_to_reg_upsample,
    ub_to_reg_downsample,
    ub_to_reg_unpack,
    ub_to_reg_unpack4,
    ub_to_reg_brcb,
    ub_to_reg_gather,
    reg_to_ub_scatter,
    gather,
    gather_mask,
)
from .dup import dup
from .group import cmax, cgmax, cmin, cgmin, cadd, cgadd, cpadd
from .interleave import deinterleave, interleave
from .mask import (
    mask_not,
    mask_and,
    mask_or,
    mask_xor,
    mask_mov,
    mask_interleave,
    mask_deinterleave,
    mask_sel,
    mask_pack,
    mask_unpack,
    move_mask_spr,
    update_mask,
)
from .unary import exp, abs, relu, sqrt, ln, log, log2, log10, neg, vnot, vcopy
from .unaryscalar import vmaxs, vmins, adds, muls, lrelu, shiftls, shiftrs, axpy
