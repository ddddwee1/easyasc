#pragma once
#include "tensorutils.h"

__aicore__ inline void constfunc_vec(GM_ADDR workspace, int M, int N, int K) {
    TPipe* pipe_ptr = GetTPipePtr();
    int _offset = 0;
    int _l0acnt = 0;
    int _l0bcnt = 0;
    int repeat = CeilDiv(Align128B(Align64B(K)) + Align16B(M) + Align256B(Align32B(N)) + Align32B(N) + Align64B(K) + GetBlockIdx() + 2*GetBlockNum() + get_subblockid(), 128);
    float _tmp_var_20 = 4.0;
    int flag_id = 1;
    SetFlag<PIPE_V, PIPE_MTE3>(flag_id);
    WaitFlag<PIPE_V, PIPE_MTE3>(flag_id);
    LocalTensor<half> ub = AllocateTensor<TPosition::VECCALC>(Align16B(M) * 1);
    Duplicate<half, false>(ub, (half)sqrt(_tmp_var_20), MASK_PLACEHOLDER, repeat, 1, 8);
}
