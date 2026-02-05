#pragma once
#include "tensorutils.h"

__aicore__ inline void constfunc_cube(GM_ADDR workspace, int M, int N, int K) {
    TPipe* pipe_ptr = GetTPipePtr();
    int _offset = 0;
    int repeat = CeilDiv(Align128B(Align64B(K)) + Align16B(M) + Align256B(Align32B(N)) + Align32B(N) + Align64B(K) + GetBlockIdx() + 2*GetBlockNum() + get_subblockid(), 128);
    float _tmp_var_16 = 4.0;
    int flag_id = 1;
    SetFlag<PIPE_M, PIPE_FIX>(flag_id);
    WaitFlag<PIPE_M, PIPE_FIX>(flag_id);
}
