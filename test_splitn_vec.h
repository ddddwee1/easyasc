#pragma once
#include "tensorutils.h"

__aicore__ inline void cubefunc_splitn_vec(GM_ADDR x_, GM_ADDR y_t_, GM_ADDR z_, GM_ADDR workspace, int M, int N, int K, int splitn) {
    TPipe* pipe_ptr = GetTPipePtr();
    int _offset = 0;
    GlobalTensor<half> x;
    x.SetGlobalBuffer((__gm__ half*) x_);
    GlobalTensor<half> y_t;
    y_t.SetGlobalBuffer((__gm__ half*) y_t_);
    GlobalTensor<float> z;
    z.SetGlobalBuffer((__gm__ float*) z_);
    DBuff<half, TPosition::A2> _l0a;
    _l0a.Init(128 * 128);
    DBuff<half, TPosition::B2> _l0b;
    _l0b.Init(128 * 128);
    int _l0acnt = 0;
    int _l0bcnt = 0;
    pipe_ptr->Reset();
    OccupyMMTE1Events();
    LocalTensor<half> l1a = AllocateTensor<TPosition::A1>(128 * K);
    LocalTensor<half> l1b = AllocateTensor<TPosition::A1>(N * K);
    LocalTensor<float> l0c = AllocateTensor<TPosition::CO1>(128 * N);
    for (int _subn = 0; _subn < N; _subn += splitn) {
        _l0bcnt = _l0bcnt + 1;
    }
    _l0acnt = _l0acnt + 1;
}
