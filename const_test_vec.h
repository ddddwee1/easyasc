#pragma once
#include "tensorutils.h"

__aicore__ inline void constfunc_vec(GM_ADDR workspace, int M, int N, int K) {
    TPipe* pipe_ptr = GetTPipePtr();
    int _offset = 0;
    DBuff<half, TPosition::A2> _l0a;
    _l0a.Init(128 * 128);
    DBuff<half, TPosition::B2> _l0b;
    _l0b.Init(128 * 128);
    pipe_ptr->Reset();
    OccupyMMTE1Events();
    int repeat = CeilDiv(((((((Align16B(M) + Align32B(N)) + Align64B(K)) + Align128B(Align64B(K))) + Align256B(Align32B(N))) + (GetBlockNum() * 2)) + GetBlockIdx()) + get_subblockid(), 128);
    float _tmp_var_20 = 4.0;
    int flag_id = 1;
    SetFlag<PIPE_V, PIPE_MTE3>(flag_id);
    WaitFlag<PIPE_V, PIPE_MTE3>(flag_id);
    LocalTensor<half> ub = AllocateTensor<TPosition::VECCALC>(Align16B(M) * 1);
    Duplicate<half, false>(ub, (half)sqrt(_tmp_var_20), MASK_PLACEHOLDER, repeat, 1, 8);
}
