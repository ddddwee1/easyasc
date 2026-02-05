#pragma once
#include "tensorutils.h"

__aicore__ inline void cubefunc_cube(GM_ADDR x_, GM_ADDR y_, GM_ADDR z_, GM_ADDR workspace, int M, int N, int K) {
    TPipe* pipe_ptr = GetTPipePtr();
    int _offset = 0;
    DEvent<PIPE_MTE2, PIPE_MTE1, false> _tmp_devent_ready_l1_0;
    DEvent<PIPE_MTE1, PIPE_MTE2, true> _tmp_devent_valid_l1_0;
    DEvent<PIPE_MTE1, PIPE_M, false> _tmp_devent_ready_l0_0;
    DEvent<PIPE_M, PIPE_MTE1, true> _tmp_devent_valid_l0_0;
    DEvent<PIPE_FIX, PIPE_M, true> _tmp_devent_valid_fix_0;
    DEvent<PIPE_M, PIPE_FIX, false> _tmp_devent_ready_fix_0;
    GlobalTensor<half> x;
    x.SetGlobalBuffer((__gm__ half*) x_);
    GlobalTensor<half> y;
    y.SetGlobalBuffer((__gm__ half*) y_);
    workspace = ShiftAddr<half>(workspace, M*N, _offset);
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
    pipe_ptr->Reset();
    OccupyMMTE1Events();
    int cnt = 0;
    int cnt1 = 0;
    int cnt2 = 0;
    int ubcnt = 0;
    float val = 1.0;
    int m_per_core = CeilDiv(M, GetBlockNum());
    int m1 = m_per_core*get_block_idx();
    int m2 = Min(m1 + m_per_core, M);
    for (int m = m1; m < m2; m += 128) {
        _tmp_devent_valid_l1_0.wait();
        GM2L1_ND2NZ(l1q.get(cnt), x[K*m], 128, K, K, 128);
        _tmp_devent_ready_l1_0.set();
        _tmp_devent_ready_l1_0.wait();
        _tmp_devent_valid_l0_0.wait();
        L0NZ2ZZ(l0a.get(cnt), l1q.get(cnt)[1024], 64, K, 128, K);
        L0NZ2NZ(l0b.get(cnt), l1k.get(cnt), 64, K, 128, K);
        _tmp_devent_ready_l0_0.set();
        _tmp_devent_ready_l0_0.wait();
        _tmp_devent_valid_fix_0.wait();
        MMAD(l0c.get(cnt), l0a.get(cnt), l0b.get(cnt), 64, K, 64, true);
        _tmp_devent_ready_fix_0.set();
        _tmp_devent_ready_fix_0.wait();
        L0C2L1(l1v, l0c.get(cnt), 128, K, 128, 128, false);
        WAIT_VEC(0, PIPE_S);
        L0C2GM_NZ2ND(y, l0c.get(cnt), 1, N, N, 128);
        cnt = cnt + 1;
        CUBE_READY(0, PIPE_FIX);
        _tmp_devent_ready_l1_0.set();
        _tmp_devent_ready_l0_0.set();
        _tmp_devent_ready_fix_0.set();
    }
    WAIT_VEC(0, PIPE_S);
    WAIT_VEC(0, PIPE_S);
}
