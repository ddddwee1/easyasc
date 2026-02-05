#pragma once
#include "tensorutils.h"

__aicore__ inline void cubefunc_vec(GM_ADDR x_, GM_ADDR y_, GM_ADDR z_, GM_ADDR workspace, int M, int N, int K) {
    TPipe* pipe_ptr = GetTPipePtr();
    int _offset = 0;
    DEvent<PIPE_V, PIPE_MTE2, true> _tmp_devent_valid_ubin_1;
    SEvent<PIPE_MTE2, PIPE_V, false> _tmp_sevent_ready_ubin_0;
    DEvent<PIPE_MTE2, PIPE_V, false> _tmp_devent_ready_ubin_1;
    SEvent<PIPE_V, PIPE_MTE2, true> _tmp_sevent_valid_ubin_0;
    DEvent<PIPE_V, PIPE_MTE3, false> _tmp_devent_ready_ubout_0;
    DEvent<PIPE_MTE3, PIPE_V, true> _tmp_devent_valid_ubout_0;
    VEC_READY(0, PIPE_MTE2);
    VEC_READY(0, PIPE_MTE2);
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
    int m1 = m_per_core*get_block_idx();
    int m2 = Min(m1 + m_per_core, M);
    for (int m = m1; m < m2; m += 128) {
        _l0acnt = _l0acnt + 1;
        _l0bcnt = _l0bcnt + 1;
        cnt = cnt + 1;
    }
    for (int m = m1; m < m2; m += 128) {
        WAIT_CUBE(0, PIPE_S);
        _tmp_sevent_valid_ubin_0.wait();
        GM2UBPAD(xubs, y[N*m], 128, 2*K, 2*K - 256, CeilDiv(0, 16));
        VEC_READY(0, PIPE_MTE2);
        _tmp_sevent_ready_ubin_0.set();
        _tmp_sevent_ready_ubin_0.wait();
        _tmp_devent_valid_ubout_0.wait();
        for (int i = 0; i < 10; i += 1) {
            _tmp_devent_valid_ubin_1.wait();
            GM2UBPAD(xub.get(cnt), x[K*m], 128, 2*K, 2*M - 256, CeilDiv(0, 16));
            _tmp_devent_ready_ubin_1.set();
            _tmp_devent_ready_ubin_1.wait();
            Add<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
            _tmp_devent_valid_ubin_1.set();
        }
        Sub<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
        Mul<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
        Div<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
        _tmp_devent_ready_ubout_0.set();
        _tmp_devent_ready_ubout_0.wait();
        UB2GMPAD(vv[N*m], xub.get(cnt), 128, 2*K, CeilDiv(0, 16), 2*M - 256);
        _tmp_sevent_valid_ubin_0.set();
        _tmp_devent_valid_ubout_0.set();
    }
    Add<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
}
