#pragma once
#include "tensorutils.h"

__aicore__ inline void cubefunc_splitn_cube(GM_ADDR x_, GM_ADDR y_t_, GM_ADDR z_, GM_ADDR workspace, int M, int N, int K, int splitn) {
    TPipe* pipe_ptr = GetTPipePtr();
    int _offset = 0;
    SEvent<PIPE_MTE1, PIPE_MTE2, true> _tmp_sevent_valid_l1_0;
    SEvent<PIPE_MTE2, PIPE_MTE1, false> _tmp_sevent_ready_l1_0;
    DEvent<PIPE_MTE1, PIPE_M, false> _tmp_devent_ready_l0_0;
    DEvent<PIPE_M, PIPE_MTE1, true> _tmp_devent_valid_l0_0;
    SEvent<PIPE_FIX, PIPE_M, true> _tmp_sevent_valid_fix_0;
    SEvent<PIPE_M, PIPE_FIX, false> _tmp_sevent_ready_fix_0;
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
    _tmp_sevent_valid_l1_0.wait();
    GM2L1_ND2NZ(l1a, x, 128, K, K, 128);
    GM2L1_ND2NZ(l1b, y_t, N, K, K, N);
    _tmp_sevent_ready_l1_0.set();
    _tmp_sevent_ready_l1_0.wait();
    _tmp_sevent_valid_fix_0.wait();
    for (int _subn = 0; _subn < N; _subn += splitn) {
        _tmp_devent_valid_l0_0.wait();
        if (_subn == 0) {
            L0NZ2ZZ(_l0a.get(_l0acnt), l1a, 128, K, 128, K);
        }
        L0NZ2NZ(_l0b.get(_l0bcnt), l1b[16*_subn], Min(N - _subn, splitn), K, N, K);
        _tmp_devent_ready_l0_0.set();
        _tmp_devent_ready_l0_0.wait();
        MMAD(l0c[_subn], _l0a.get(_l0acnt), _l0b.get(_l0bcnt), 128, K, Min(N - _subn, splitn), true);
        _l0bcnt = _l0bcnt + 1;
        _tmp_devent_valid_l0_0.set();
    }
    _l0acnt = _l0acnt + 1;
    _tmp_sevent_ready_fix_0.set();
    _tmp_sevent_ready_fix_0.wait();
    L0C2GM_NZ2ND(z, l0c, 128, N, N, 128);
    _tmp_sevent_ready_l1_0.set();
    _tmp_sevent_ready_fix_0.set();
}
