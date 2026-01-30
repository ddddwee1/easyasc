DEvent<PIPE_V, PIPE_MTE2> _tmp_event1_ubin_valid;
DEvent<PIPE_MTE2, PIPE_V> _tmp_event1_ubin_ready;
DEvent<PIPE_V, PIPE_MTE2> _tmp_event2_ubin_valid;
DEvent<PIPE_MTE2, PIPE_V> _tmp_event2_ubin_ready;
GlobalTensor<half> x;
x.SetGlobalBuffer((__gm__ half*) x_);
GlobalTensor<half> z;
z.SetGlobalBuffer((__gm__ half*) z_);
DBuff<half, TPosition::VECCALC> xub;
LocalTensor<half> xubs;
int cnt = 0;
int cnt1 = 0;
int cnt2 = 0;
int m_per_core = CeilDiv(M, GetBlockNum());
int m1 = m_per_core*get_block_idx();
int m2 = Min(m1 + m_per_core, M);
for (int m = 0; m < m2; m += 128) {
    // start auto sync
    cnt = cnt + 1;
    // end auto sync
}
// start auto sync
for (int m = 0; m < m2; m += 128) {
    _tmp_event1_ubin_valid.wait();
    GM2UBPAD(xub.get(cnt), x[K*m], 128, 2*K, 2*M - 256, CeilDiv(0, 16));
    _tmp_event1_ubin_ready.set();
    _tmp_event1_ubin_ready.wait();
    Add<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
    Sub<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
    Mul<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
    Div<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
    _tmp_event1_ubin_valid.set();
    UB2GMPAD(z[N*m], xub.get(cnt), 128, 2*K, CeilDiv(0, 16), 2*M - 256);
}
// end auto sync
// start auto sync
for (int m = 0; m < m2; m += 128) {
    _tmp_event1_ubin_valid.wait();
    GM2UBPAD(xub.get(cnt), x[K*m], 128, 2*K, 2*M - 256, CeilDiv(0, 16));
    _tmp_event1_ubin_ready.set();
    _tmp_event1_ubin_ready.wait();
    for (int i = 0; i < 10; i += 1) {
        Add<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
    }
    Sub<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
    Mul<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
    Div<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
    _tmp_event1_ubin_valid.set();
    UB2GMPAD(z[N*m], xub.get(cnt), 128, 2*K, CeilDiv(0, 16), 2*M - 256);
}
// end auto sync
// start auto sync
for (int m = 0; m < m2; m += 128) {
    _tmp_event1_ubin_valid.wait();
    GM2UBPAD(xub.get(cnt), x[K*m], 128, 2*K, 2*M - 256, CeilDiv(0, 16));
    _tmp_event1_ubin_ready.set();
    _tmp_event1_ubin_ready.wait();
    for (int i = 0; i < 10; i += 1) {
        Add<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
        Sub<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
        Mul<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
        Div<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
        UB2GMPAD(z[N*m], xub.get(cnt), 128, 2*K, CeilDiv(0, 16), 2*M - 256);
    }
    _tmp_event1_ubin_valid.set();
}
// end auto sync
// start auto sync
for (int m = 0; m < m2; m += 128) {
    for (int i = 0; i < 10; i += 1) {
        _tmp_event1_ubin_valid.wait();
        GM2UBPAD(xub.get(cnt), x[K*m], 128, 2*K, 2*M - 256, CeilDiv(0, 16));
        _tmp_event1_ubin_ready.set();
        _tmp_event1_ubin_ready.wait();
        Add<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
        Sub<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
        Mul<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
        Div<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
        _tmp_event1_ubin_valid.set();
    }
    UB2GMPAD(z[N*m], xub.get(cnt), 128, 2*K, CeilDiv(0, 16), 2*M - 256);
}
// end auto sync
// start auto sync
for (int m = 0; m < m2; m += 128) {
    for (int i = 0; i < 10; i += 1) {
        _tmp_event1_ubin_valid.wait();
        GM2UBPAD(xub.get(cnt), x[K*m], 128, 2*K, 2*M - 256, CeilDiv(0, 16));
        _tmp_event1_ubin_ready.set();
        _tmp_event1_ubin_ready.wait();
        Add<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
    }
    Sub<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
    Mul<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
    Div<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
    _tmp_event1_ubin_valid.set();
    UB2GMPAD(z[N*m], xub.get(cnt), 128, 2*K, CeilDiv(0, 16), 2*M - 256);
}
// end auto sync
// start auto sync
for (int m = 0; m < m2; m += 128) {
    _tmp_event1_ubin_valid.wait();
    GM2UBPAD(xub.get(cnt), x[K*m], 128, 2*K, 2*M - 256, CeilDiv(0, 16));
    _tmp_event1_ubin_ready.set();
    _tmp_event1_ubin_ready.wait();
    for (int i = 0; i < 10; i += 1) {
        _tmp_event2_ubin_valid.wait();
        GM2UBPAD(xub.get(cnt), x[K*m], 128, 2*K, 2*M - 256, CeilDiv(0, 16));
        _tmp_event2_ubin_ready.set();
        _tmp_event2_ubin_ready.wait();
        Add<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
        _tmp_event2_ubin_valid.set();
    }
    Sub<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
    Mul<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
    Div<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
    _tmp_event1_ubin_valid.set();
    UB2GMPAD(z[N*m], xub.get(cnt), 128, 2*K, CeilDiv(0, 16), 2*M - 256);
}
// end auto sync
// start auto sync
for (int m = 0; m < m2; m += 128) {
    _tmp_event1_ubin_valid.wait();
    GM2UBPAD(xub.get(cnt), x[K*m], 128, 2*K, 2*M - 256, CeilDiv(0, 16));
    _tmp_event1_ubin_ready.set();
    _tmp_event1_ubin_ready.wait();
    for (int i = 0; i < 10; i += 1) {
        _tmp_event2_ubin_valid.wait();
        GM2UBPAD(xub.get(cnt), x[K*m], 128, 2*K, 2*M - 256, CeilDiv(0, 16));
        _tmp_event2_ubin_ready.set();
        _tmp_event2_ubin_ready.wait();
        Add<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
        _tmp_event2_ubin_valid.set();
    }
    Sub<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
    Mul<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
    Div<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
    _tmp_event1_ubin_valid.set();
    UB2GMPAD(z[N*m], xub.get(cnt), 128, 2*K, CeilDiv(0, 16), 2*M - 256);
}
// end auto sync
// start auto sync
for (int m = 0; m < m2; m += 128) {
    _tmp_event1_ubin_valid.wait();
    GM2UBPAD(xubs, x[K*m], 128, 2*K, 2*M - 256, CeilDiv(0, 16));
    _tmp_event1_ubin_ready.set();
    _tmp_event1_ubin_ready.wait();
    for (int i = 0; i < 10; i += 1) {
        _tmp_event2_ubin_valid.wait();
        GM2UBPAD(xub.get(cnt), x[K*m], 128, 2*K, 2*M - 256, CeilDiv(0, 16));
        _tmp_event2_ubin_ready.set();
        _tmp_event2_ubin_ready.wait();
        Add<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
        _tmp_event2_ubin_valid.set();
    }
    Sub<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
    Mul<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
    Div<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
    _tmp_event1_ubin_valid.set();
    UB2GMPAD(z[N*m], xub.get(cnt), 128, 2*K, CeilDiv(0, 16), 2*M - 256);
}
// end auto sync
// start auto sync
for (int m = 0; m < m2; m += 128) {
    _tmp_event1_ubin_valid.wait();
    GM2UBPAD(xub.get(cnt), x[K*m], 128, 2*K, 2*M - 256, CeilDiv(0, 16));
    _tmp_event1_ubin_ready.set();
    _tmp_event1_ubin_ready.wait();
    for (int i = 0; i < 10; i += 1) {
        _tmp_event2_ubin_valid.wait();
        GM2UBPAD(xubs, x[K*m], 128, 2*K, 2*M - 256, CeilDiv(0, 16));
        _tmp_event2_ubin_ready.set();
        _tmp_event2_ubin_ready.wait();
        Add<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
        _tmp_event2_ubin_valid.set();
    }
    Sub<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
    Mul<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
    Div<half, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 128), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
    _tmp_event1_ubin_valid.set();
    UB2GMPAD(z[N*m], xub.get(cnt), 128, 2*K, CeilDiv(0, 16), 2*M - 256);
}
// end auto sync
