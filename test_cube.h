#pragma once
#include "tensorutils.h"

__aicore__ inline void cubefunc_cube(GM_ADDR x_, GM_ADDR y_, GM_ADDR z_, GM_ADDR workspace, int M, int N, int K) {
    TPipe* pipe_ptr = GetTPipePtr();
    int _offset = 0;
    DEvent<PIPE_MTE2, PIPE_MTE1, false> _tmp_devent_ready_l1_0;
    DEvent<PIPE_MTE1, PIPE_MTE2, true> _tmp_devent_valid_l1_0;
    DEvent<PIPE_MTE1, PIPE_M, false> _tmp_devent_ready_l0_0;
    DEvent<PIPE_M, PIPE_MTE1, true> _tmp_devent_valid_l0_0;
    DEvent<PIPE_M, PIPE_FIX, false> _tmp_devent_ready_fix_0;
    DEvent<PIPE_FIX, PIPE_M, true> _tmp_devent_valid_fix_0;
    GlobalTensor<half> x;
    x.SetGlobalBuffer((__gm__ half*) x_);
    GlobalTensor<half> y;
    y.SetGlobalBuffer((__gm__ half*) y_);
    DBuff<half, TPosition::A2> _l0a;
    _l0a.Init(128 * 128);
    DBuff<half, TPosition::B2> _l0b;
    _l0b.Init(128 * 128);
    int _l0acnt = 0;
    int _l0bcnt = 0;
    pipe_ptr->Reset();
    OccupyMMTE1Events();
    workspace = ShiftAddr<half>(workspace, M * N, _offset);
    GlobalTensor<half> vv;
    vv.SetGlobalBuffer((__gm__ half*) workspace);
    DBuff<half, TPosition::A1> l1q;
    l1q.Init(128 * K);
    DBuff<half, TPosition::A1> l1k;
    l1k.Init(128 * K);
    LocalTensor<half> l1v = AllocateTensor<TPosition::A1>(128 * K);
    DBuff<half, TPosition::A2> l0a;
    l0a.Init(128 * K);
    DBuff<half, TPosition::B2> l0b;
    l0b.Init(128 * K);
    DBuff<float, TPosition::CO1> l0c;
    l0c.Init(128 * K);
    LocalTensor<float> l0c2 = AllocateTensor<TPosition::CO1>(128 * K);
    DBuff<half, TPosition::VECCALC> xub;
    xub.Init(128 * K);
    pipe_ptr->Reset();
    OccupyMMTE1Events();
    LocalTensor<half> xubs = AllocateTensor<TPosition::VECCALC>(128 * K);
    int cnt = 0;
    int cnt1 = 0;
    int cnt2 = 0;
    int m_per_core = CeilDiv(M, GetBlockNum());
    int m1 = m_per_core * get_block_idx();
    int m2 = Min(m1 + m_per_core, M);
    for (int m = m1; m < m2; m += 128) {
        _tmp_devent_valid_l1_0.wait();
        GM2L1_ND2NZ(l1q.get(cnt), x[m * K + 0], (m + 128) - m, K - 0, K, 128);
        _tmp_devent_ready_l1_0.set();
        _tmp_devent_ready_l1_0.wait();
        _tmp_devent_valid_l0_0.wait();
        L0NZ2ZZ(_l0a.get(_l0acnt), l1q.get(cnt)[64 * 16 + 0 * 128], 64, K - 0, 128, K);
        L0NZ2NZ(_l0b.get(_l0bcnt), l1k.get(cnt)[0 * 16 + 0 * 128], 64, K - 0, 128, K);
        _tmp_devent_ready_l0_0.set();
        _tmp_devent_ready_l0_0.wait();
        _tmp_devent_valid_fix_0.wait();
        MMAD(l0c.get(cnt), _l0a.get(_l0acnt), _l0b.get(_l0bcnt), 64, K - 0, 64, true);
        _l0acnt = _l0acnt + 1;
        _l0bcnt = _l0bcnt + 1;
        _tmp_devent_ready_fix_0.set();
        _tmp_devent_ready_fix_0.wait();
        L0C2L1(l1v, l0c.get(cnt), 128, K, 128, 128, false);
        WAIT_VEC(0, PIPE_S);
        L0C2GM_NZ2ND(y[0 * N + 0], l0c.get(cnt), 1, N - 0, N, 128);
        cnt = cnt + 1;
        CUBE_READY(0, PIPE_FIX);
        _tmp_devent_ready_l1_0.set();
        _tmp_devent_ready_l0_0.set();
        _tmp_devent_ready_fix_0.set();
    }
    WAIT_VEC(0, PIPE_S);
    WAIT_VEC(0, PIPE_S);
}
