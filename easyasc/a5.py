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
from .flowcontrol import range, unroll, If, Elif, Else
from .torchplutin import OpExec
from .shortcuts import matmul

from . import globvars 
globvars.device_type = 'david'
