from .utils.datatype import Datatype as DT
from .utils.Tensor import Tensor, DBuff, GMTensor
from .utils.var import Var
from .decorators import kernel, func, auto_sync, vf
from .utils.positions import Position, PositionType
from .utils.roundmode import RoundMode, RoundModeType
from .utils.vecop import maximum, minimum
from .utils.comparemode import CompareMode, CompareModeType
from .utils.selectmode import SelectMode, SelectModeType
from .utils.pipe import Pipe, PipeType
from .utils.events import SEvent, DEvent
from .utils.mutex import CvMutex, VcMutex
from .utils.reg import Reg, MaskReg, MaskType
from .utils.castconfig import CastConfig
from .stub_functions import (
    CeilDiv, GetCubeNum, GetCubeIdx, GetVecNum, GetVecIdx, GetSubBlockIdx,  # stub_functions/var_op.py
    scalar_sqrt,                                                            # stub_functions/var_op.py
    Align16, Align32, Align64, Align128, Align256,                          # stub_functions/var_op.py
    var_mul, var_add, var_sub, var_div, Min, Max,                           # stub_functions/var_op.py
    gm_to_l1_nd2nz, l1_to_l0, mmad, l0c_to_gm_nz2nd, l0c_to_l1,             # stub_functions/cube.py
    bar_m, bar_v, bar_mte3, bar_mte2, bar_mte1, bar_fix, bar_all,           # stub_functions/barrier.py
    atomic_add, atomic_max, atomic_min,                                     # stub_functions/atomic.py
    reinterpret, split_workspace, reset_cache,                              # stub_functions/misc.py
    cube_ready, vec_ready, wait_cube, wait_vec,                             # stub_functions/crosscore.py
    allcube_ready, allvec_ready, allcube_wait, allvec_wait,                 # stub_functions/crosscore.py
    setflag, waitflag,                                                      # stub_functions/flags.py
)
from .stub_functions.micro import (
    arange,                                                                 # stub_functions/micro/arange.py
    vmax, vmin, add, sub, mul, div, vand, vor, vxor, prelu,                 # stub_functions/micro/binary.py
    cast,                                                                   # stub_functions/micro/cast.py
    compare, select,                                                        # stub_functions/micro/compare.py
    LoadDist, LoadDistValue, StoreDist, StoreDistValue,                     # stub_functions/micro/datamove.py
    ub_to_reg, reg_to_ub, ub_to_reg_continuous, reg_to_ub_continuous,       # stub_functions/micro/datamove.py
    reg_to_ub_downsample, reg_to_ub_pack4, reg_to_ub_single,                # stub_functions/micro/datamove.py
    ub_to_reg_single, ub_to_reg_upsample, ub_to_reg_downsample,             # stub_functions/micro/datamove.py
    ub_to_reg_unpack, ub_to_reg_unpack4, ub_to_reg_brcb,                    # stub_functions/micro/datamove.py
    ub_to_reg_gather, reg_to_ub_scatter, gather, gather_mask,               # stub_functions/micro/datamove.py
    dup,                                                                    # stub_functions/micro/dup.py
    cmax, cgmax, cmin, cgmin, cadd, cgadd, cpadd,                           # stub_functions/micro/group.py
    deinterleave, interleave,                                               # stub_functions/micro/interleave.py
    mask_not, mask_and, mask_or, mask_xor, mask_mov, mask_interleave,        # stub_functions/micro/mask.py
    mask_deinterleave, mask_sel, mask_pack, mask_unpack,                    # stub_functions/micro/mask.py
    move_mask_spr, update_mask,                                             # stub_functions/micro/mask.py
    exp, abs, relu, sqrt, ln, log, log2, log10, neg, vnot, vcopy,           # stub_functions/micro/unary.py
    vmaxs, vmins, adds, muls, lrelu, shiftls, shiftrs, axpy,                # stub_functions/micro/unaryscalar.py
)
from .stub_functions import micro
from .flowcontrol import range, unroll, If, Elif, Else
from .torchplutin import OpExec
from .shortcuts import matmul

from . import globvars 
globvars.device_type = 'david'
