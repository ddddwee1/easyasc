SEvent<PIPE_MTE2, PIPE_V> _tmp_sevent_ready_ubin_0;
DEvent<PIPE_MTE2, PIPE_V> _tmp_devent_ready_ubin_1;
DEvent<PIPE_V, PIPE_MTE2> _tmp_devent_valid_ubin_0;
DEvent<PIPE_MTE2, PIPE_V> _tmp_devent_ready_ubin_0;
DEvent<PIPE_V, PIPE_MTE2> _tmp_devent_valid_ubin_1;
SEvent<PIPE_V, PIPE_MTE2> _tmp_sevent_valid_ubin_0;
DEvent<PIPE_V, PIPE_MTE3> _tmp_devent_ready_ubout_0;
DEvent<PIPE_MTE3, PIPE_V> _tmp_devent_valid_ubout_0;
GlobalTensor<half> x;
x.SetGlobalBuffer((__gm__ half*) x_);
GlobalTensor<half> y;
y.SetGlobalBuffer((__gm__ half*) y_);
GlobalTensor<half> z;
z.SetGlobalBuffer((__gm__ half*) z_);
workspace = ShiftAddr<half>(workspace, M*N, offset);
GlobalTensor<half> vv;
vv.SetGlobalBuffer((__gm__ half*) workspace);
DBuff<half, TPosition::VECCALC> xub;
LocalTensor<half> xubs;
int cnt = 0;
int cnt1 = 0;
int cnt2 = 0;
int ubcnt = 0;
float val = 1.0;
int m_per_core = CeilDiv(M, GetBlockNum());
int m1 = m_per_core*get_block_idx();
int m2 = Min(m1 + m_per_core, M);
for (int m = m1; m < m2; m += 128) {
    cnt = cnt + 1;
}
for (int m = m1; m < m2; m += 128) {
    _tmp_devent_valid_ubin_0.wait();
    GM2UBPAD(xub.get(cnt), x[K*m], 128, 2*K, 2*M - 256, CeilDiv(0, 16));
    _tmp_devent_ready_ubin_0.set();
    _tmp_devent_ready_ubin_0.wait();
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
    UB2GMPAD(z[N*m], xub.get(cnt), 128, 2*K, CeilDiv(0, 16), 2*M - 256);
    _tmp_devent_valid_ubin_0.set();
    _tmp_devent_valid_ubout_0.set();
}
for (int m = m1; m < m2; m += 128) {
    _tmp_sevent_valid_ubin_0.wait();
    GM2UBPAD(xubs, x[K*m], 128, 2*K, 2*M - 256, CeilDiv(0, 16));
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
    UB2GMPAD(z[N*m], xub.get(cnt), 128, 2*K, CeilDiv(0, 16), 2*M - 256);
    _tmp_sevent_valid_ubin_0.set();
    _tmp_devent_valid_ubout_0.set();
}
for (int m = m1; m < m2; m += 128) {
    _tmp_sevent_valid_ubin_0.wait();
    GM2UBPAD(xubs, x[K*m], 128, 2*K, 2*M - 256, CeilDiv(0, 16));
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
    UB2GMPAD(z[N*m], xub.get(cnt), 128, 2*K, CeilDiv(0, 16), 2*M - 256);
    _tmp_sevent_valid_ubin_0.set();
    _tmp_devent_valid_ubout_0.set();
}
for (int m = m1; m < m2; m += 128) {
    _tmp_sevent_valid_ubin_0.wait();
    GM2UBPAD(xubs, x[K*m], 128, 2*K, 2*M - 256, CeilDiv(0, 16));
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
