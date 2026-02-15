from .utils.datatype import Datatype as DT
from .utils.Tensor import Tensor, DBuff, GMTensor
from .utils.var import Var
from .decorators import kernel, func, auto_sync
from .utils.positions import Position, PositionType
from .utils.roundmode import RoundMode, RoundModeType
from .utils.vecop import maximum, minimum
from .utils.comparemode import CompareMode, CompareModeType
from .utils.selectmode import SelectMode, SelectModeType
from .utils.pipe import Pipe, PipeType
from .utils.events import SEvent, DEvent
from .utils.mutex import CvMutex, VcMutex
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
    add, sub, mul, div, vmax, vmin, vand, vor, muladddst,                   # stub_functions/vec/binary.py
    exp, ln, abs, rec, sqrt, rsqrt, vnot, relu,                             # stub_functions/vec/unary.py
    adds, muls, vmaxs, vmins, lrelu, axpy,                                  # stub_functions/vec/unaryscalar.py
    dup, brcb,                                                              # stub_functions/vec/dupbrcb.py
    gather, scatter,                                                        # stub_functions/vec/gatherscatter.py
    cmax, cgmax, cmin, cgmin, cadd, cgadd, cpadd,                           # stub_functions/vec/group.py
    sort32, mergesort4, mergesort_2seq,                                     # stub_functions/vec/sort.py
    set_mask, reset_mask,                                                   # stub_functions/vec/vecmask.py
    gm_to_ub_pad, ub_to_gm_pad, ub_to_ub,                                   # stub_functions/vec/datamove.py
    cast,                                                                   # stub_functions/vec/cast.py
    compare, compare_scalar, set_cmpmask,                                   # stub_functions/vec/compare.py
    select,                                                                 # stub_functions/vec/select.py
)
from .flowcontrol import range, unroll, If, Elif, Else
from .torchplutin import OpExec
from .shortcuts import matmul

from . import globvars 
globvars.device_type = 'b3'
