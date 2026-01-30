DEvent<PIPE_FIX, PIPE_M> _tmp_event1_fix_valid;
DEvent<PIPE_M, PIPE_FIX> _tmp_event1_fix_ready;
SEvent<PIPE_FIX, PIPE_M> _tmp_sevent1_fix_valid;
SEvent<PIPE_M, PIPE_FIX> _tmp_sevent1_fix_ready;
DEvent<PIPE_M, PIPE_MTE1> _tmp_event1_l0_valid;
DEvent<PIPE_MTE1, PIPE_M> _tmp_event1_l0_ready;
DEvent<PIPE_MTE1, PIPE_MTE2> _tmp_event1_l1_valid;
DEvent<PIPE_MTE2, PIPE_MTE1> _tmp_event1_l1_ready;
SEvent<PIPE_MTE1, PIPE_MTE2> _tmp_sevent1_l1_valid;
SEvent<PIPE_MTE2, PIPE_MTE1> _tmp_sevent1_l1_ready;
DEvent<PIPE_MTE1, PIPE_MTE2> _tmp_event2_l1_valid;
DEvent<PIPE_MTE2, PIPE_MTE1> _tmp_event2_l1_ready;
DEvent<PIPE_MTE1, PIPE_MTE2> _tmp_event3_l1_valid;
DEvent<PIPE_MTE2, PIPE_MTE1> _tmp_event3_l1_ready;
GlobalTensor<half> x;
x.SetGlobalBuffer((__gm__ half*) x_);
GlobalTensor<half> y;
y.SetGlobalBuffer((__gm__ half*) y_);
DBuff<half, TPosition::A1> l1q;
DBuff<half, TPosition::A1> l1k;
LocalTensor<half> l1v;
DBuff<half, TPosition::A2> l0a;
DBuff<half, TPosition::B2> l0b;
DBuff<float, TPosition::CO1> l0c;
LocalTensor<float> l0c2;
int cnt = 0;
int m_per_core = CeilDiv(M, GetBlockNum());
int m1 = m_per_core*get_block_idx();
int m2 = Min(m1 + m_per_core, M);
for (int m = 0; m < m2; m += 128) {
    // start auto sync
    _tmp_event1_l1_valid.wait();
    GM2L1_ND2NZ(l1q.get(cnt), x[K*m], 128, K, K, 128);
    _tmp_event1_l1_ready.set();
    _tmp_event1_l1_ready.wait();
    _tmp_event1_l0_valid.wait();
    L0NZ2ZZ(l0a.get(cnt), l1q.get(cnt)[1024], 64, K, 128, K);
    L0NZ2NZ(l0b.get(cnt), l1k.get(cnt), 64, K, 128, K);
    _tmp_event1_l0_ready.set();
    _tmp_event1_l0_ready.wait();
    _tmp_event1_l1_valid.set();
    _tmp_event1_fix_valid.wait();
    MMAD(l0c.get(cnt), l0a.get(cnt), l0b.get(cnt), 64, K, 64, true);
    _tmp_event1_fix_ready.set();
    _tmp_event1_fix_ready.wait();
    _tmp_event1_l0_valid.set();
    L0C2GM_NZ2ND(y, l0c.get(cnt), 1, N, N, 128);
    _tmp_event1_fix_valid.set();
    cnt = cnt + 1;
    // end auto sync
}
for (int m = 0; m < m2; m += 128) {
    // start auto sync
    _tmp_event1_l1_valid.wait();
    GM2L1_ND2NZ(l1q.get(cnt), x[K*m], 128, K, K, 128);
    _tmp_event1_l1_ready.set();
    _tmp_event1_l1_ready.wait();
    int j = 0;
    _tmp_event1_fix_valid.wait();
    for (; j < 10; j += 1) {
        _tmp_event1_l0_valid.wait();
        L0NZ2ZZ(l0a.get(cnt), l1q.get(cnt)[1024], 64, K, 128, K);
        L0NZ2NZ(l0b.get(cnt), l1k.get(cnt), 64, K, 128, K);
        _tmp_event1_l0_ready.set();
        _tmp_event1_l0_ready.wait();
        MMAD(l0c.get(cnt), l0a.get(cnt), l0b.get(cnt), 64, K, 64, true);
        _tmp_event1_l0_valid.set();
    }
    _tmp_event1_fix_ready.set();
    _tmp_event1_fix_ready.wait();
    _tmp_event1_l1_valid.set();
    L0C2GM_NZ2ND(y, l0c.get(cnt), 1, N, N, 128);
    _tmp_event1_fix_valid.set();
    cnt = cnt + 1;
    // end auto sync
}
for (int m = 0; m < m2; m += 128) {
    // start auto sync
    _tmp_event1_l1_valid.wait();
    GM2L1_ND2NZ(l1q.get(cnt), x[K*m], 128, K, K, 128);
    GM2L1_ND2NZ(l1k.get(cnt), x[K*m], 128, K, K, 128);
    _tmp_event1_l1_ready.set();
    _tmp_event1_l1_ready.wait();
    int j = 0;
    _tmp_event1_l0_valid.wait();
    for (; j < 10; j += 1) {
        L0NZ2ZZ(l0a.get(cnt), l1q.get(cnt)[1024], 64, K, 128, K);
        L0NZ2NZ(l0b.get(cnt), l1k.get(cnt), 64, K, 128, K);
    }
    _tmp_event1_l0_ready.set();
    _tmp_event1_l0_ready.wait();
    _tmp_event1_l1_valid.set();
    _tmp_event1_fix_valid.wait();
    MMAD(l0c.get(cnt), l0a.get(cnt), l0b.get(cnt), 64, K, 64, true);
    _tmp_event1_fix_ready.set();
    _tmp_event1_fix_ready.wait();
    _tmp_event1_l0_valid.set();
    L0C2GM_NZ2ND(y, l0c.get(cnt), 1, N, N, 128);
    _tmp_event1_fix_valid.set();
    cnt = cnt + 1;
    // end auto sync
}
for (int m = 0; m < m2; m += 128) {
    // start auto sync
    _tmp_event1_l1_valid.wait();
    GM2L1_ND2NZ(l1q.get(cnt), x[K*m], 128, K, K, 128);
    _tmp_event1_l1_ready.set();
    _tmp_event1_l1_ready.wait();
    int j = 0;
    _tmp_event1_l0_valid.wait();
    for (; j < 10; j += 1) {
        _tmp_event2_l1_valid.wait();
        GM2L1_ND2NZ(l1k.get(cnt), x[K*m], 128, K, K, 128);
        _tmp_event2_l1_ready.set();
        _tmp_event2_l1_ready.wait();
        L0NZ2ZZ(l0a.get(cnt), l1q.get(cnt)[1024], 64, K, 128, K);
        L0NZ2NZ(l0b.get(cnt), l1k.get(cnt), 64, K, 128, K);
        _tmp_event2_l1_valid.set();
    }
    _tmp_event1_l0_ready.set();
    _tmp_event1_l0_ready.wait();
    _tmp_event1_l1_valid.set();
    _tmp_event1_fix_valid.wait();
    MMAD(l0c.get(cnt), l0a.get(cnt), l0b.get(cnt), 64, K, 64, true);
    _tmp_event1_fix_ready.set();
    _tmp_event1_fix_ready.wait();
    _tmp_event1_l0_valid.set();
    L0C2GM_NZ2ND(y, l0c.get(cnt), 1, N, N, 128);
    _tmp_event1_fix_valid.set();
    cnt = cnt + 1;
    // end auto sync
}
// start auto sync
_tmp_event1_l1_valid.wait();
GM2L1_ND2NZ(l1q.get(cnt), x[K*m], 128, K, K, 128);
_tmp_event1_l1_ready.set();
_tmp_event1_l1_ready.wait();
for (int m = 0; m < m2; m += 128) {
    _tmp_event2_l1_valid.wait();
    GM2L1_ND2NZ(l1q.get(cnt), x[K*m], 128, K, K, 128);
    _tmp_event2_l1_ready.set();
    _tmp_event2_l1_ready.wait();
    for (int j = 0; j < 10; j += 1) {
        _tmp_event3_l1_valid.wait();
        GM2L1_ND2NZ(l1q.get(cnt), x[K*m], 128, K, K, 128);
        _tmp_event3_l1_ready.set();
        _tmp_event3_l1_ready.wait();
        int k = 0;
        _tmp_event1_fix_valid.wait();
        for (; k < 20; k += 1) {
            _tmp_event1_l0_valid.wait();
            L0NZ2ZZ(l0a.get(cnt), l1q.get(cnt)[1024], 64, K, 128, K);
            L0NZ2NZ(l0b.get(cnt), l1k.get(cnt), 64, K, 128, K);
            _tmp_event1_l0_ready.set();
            _tmp_event1_l0_ready.wait();
            MMAD(l0c.get(cnt), l0a.get(cnt), l0b.get(cnt), 64, K, 64, true);
            _tmp_event1_l0_valid.set();
        }
        _tmp_event1_fix_ready.set();
        _tmp_event1_fix_ready.wait();
        _tmp_event3_l1_valid.set();
        L0C2GM_NZ2ND(y, l0c.get(cnt), 1, N, N, 128);
        _tmp_event1_fix_valid.set();
    }
    _tmp_event2_l1_valid.set();
}
_tmp_event1_l1_valid.set();
// end auto sync
// start auto sync
for (int m = 0; m < m2; m += 128) {
    _tmp_event1_l1_valid.wait();
    GM2L1_ND2NZ(l1q.get(cnt), x[K*m], 128, K, K, 128);
    _tmp_event1_l1_ready.set();
    _tmp_event1_l1_ready.wait();
    for (int j = 0; j < 10; j += 1) {
        _tmp_event2_l1_valid.wait();
        if (m == 0) {
            GM2L1_ND2NZ(l1q.get(cnt), x[K*m], 128, K, K, 128);
        }
        _tmp_event2_l1_ready.set();
        _tmp_event2_l1_ready.wait();
        for (int k = 0; k < 20; k += 1) {
            _tmp_event1_l0_valid.wait();
            L0NZ2ZZ(l0a.get(cnt), l1q.get(cnt)[1024], 64, K, 128, K);
            L0NZ2NZ(l0b.get(cnt), l1k.get(cnt), 64, K, 128, K);
            _tmp_event1_l0_ready.set();
            _tmp_event1_l0_ready.wait();
            _tmp_event1_fix_valid.wait();
            if (cnt == 0) {
                MMAD(l0c.get(cnt), l0a.get(cnt), l0b.get(cnt), 64, K, 64, true);
            } else {
                MMAD(l0c.get(cnt), l0a.get(cnt), l0b.get(cnt), 64, K, 64, false);
            }
            _tmp_event1_fix_ready.set();
            _tmp_event1_fix_ready.wait();
            _tmp_event1_l0_valid.set();
            if (k == 1) {
                L0C2GM_NZ2ND(y, l0c.get(cnt), 1, N, N, 128);
            }
            _tmp_event1_fix_valid.set();
        }
        _tmp_event2_l1_valid.set();
    }
    _tmp_event1_l1_valid.set();
}
// end auto sync
// start auto sync
for (int m = 0; m < m2; m += 128) {
    _tmp_event1_l1_valid.wait();
    GM2L1_ND2NZ(l1q.get(cnt), x[K*m], 128, K, K, 128);
    _tmp_event1_l1_ready.set();
    _tmp_event1_l1_ready.wait();
    for (int j = 0; j < 10; j += 1) {
        _tmp_event2_l1_valid.wait();
        if (m == 0) {
            GM2L1_ND2NZ(l1q.get(cnt), x[K*m], 128, K, K, 128);
        }
        _tmp_event2_l1_ready.set();
        _tmp_event2_l1_ready.wait();
        for (int k = 0; k < 20; k += 1) {
            _tmp_event1_l0_valid.wait();
            L0NZ2ZZ(l0a.get(cnt), l1q.get(cnt)[1024], 64, K, 128, K);
            L0NZ2NZ(l0b.get(cnt), l1k.get(cnt), 64, K, 128, K);
            _tmp_event1_l0_ready.set();
            _tmp_event1_l0_ready.wait();
            MMAD(l0c.get(cnt), l0a.get(cnt), l0b.get(cnt), 64, K, 64, true);
            _tmp_event1_l0_valid.set();
            _tmp_event1_l0_valid.wait();
            L0NZ2ZZ(l0a.get(cnt), l1q.get(cnt)[1024], 64, K, 128, K);
            L0NZ2NZ(l0b.get(cnt), l1k.get(cnt), 64, K, 128, K);
            _tmp_event1_l0_ready.set();
            _tmp_event1_l0_ready.wait();
            MMAD(l0c.get(cnt), l0a.get(cnt), l0b.get(cnt), 64, K, 64, true);
            _tmp_event1_l0_valid.set();
        }
        _tmp_event2_l1_valid.set();
    }
    _tmp_event1_l1_valid.set();
}
// end auto sync
// start auto sync
for (int m = 0; m < m2; m += 128) {
    _tmp_event1_l1_valid.wait();
    GM2L1_ND2NZ(l1q.get(cnt), x[K*m], 128, K, K, 128);
    _tmp_event1_l1_ready.set();
    _tmp_event1_l1_ready.wait();
    for (int j = 0; j < 10; j += 1) {
        _tmp_event2_l1_valid.wait();
        if (m == 0) {
            GM2L1_ND2NZ(l1q.get(cnt), x[K*m], 128, K, K, 128);
        }
        _tmp_event2_l1_ready.set();
        _tmp_event2_l1_ready.wait();
        for (int k = 0; k < 20; k += 1) {
            _tmp_event1_l0_valid.wait();
            L0NZ2ZZ(l0a.get(cnt), l1q.get(cnt)[1024], 64, K, 128, K);
            L0NZ2NZ(l0b.get(cnt), l1k.get(cnt), 64, K, 128, K);
            _tmp_event1_l0_ready.set();
            _tmp_event1_l0_ready.wait();
            _tmp_event1_fix_valid.wait();
            MMAD(l0c.get(cnt), l0a.get(cnt), l0b.get(cnt), 64, K, 64, true);
            _tmp_event1_fix_ready.set();
            _tmp_event1_fix_ready.wait();
            _tmp_event1_l0_valid.set();
            L0C2GM_NZ2ND(y, l0c.get(cnt), 1, N, N, 128);
            _tmp_event1_fix_valid.set();
            _tmp_event1_l0_valid.wait();
            L0NZ2ZZ(l0a.get(cnt), l1q.get(cnt)[1024], 64, K, 128, K);
            L0NZ2NZ(l0b.get(cnt), l1k.get(cnt), 64, K, 128, K);
            _tmp_event1_l0_ready.set();
            _tmp_event1_l0_ready.wait();
            _tmp_event1_fix_valid.wait();
            MMAD(l0c.get(cnt), l0a.get(cnt), l0b.get(cnt), 64, K, 64, true);
            _tmp_event1_fix_ready.set();
            _tmp_event1_fix_ready.wait();
            _tmp_event1_l0_valid.set();
            L0C2GM_NZ2ND(y, l0c.get(cnt), 1, N, N, 128);
            _tmp_event1_fix_valid.set();
        }
        _tmp_event2_l1_valid.set();
    }
    _tmp_event1_l1_valid.set();
}
// end auto sync
// start auto sync
for (int m = 0; m < m2; m += 128) {
    _tmp_event1_l1_valid.wait();
    GM2L1_ND2NZ(l1q.get(cnt), x[K*m], 128, K, K, 128);
    _tmp_event1_l1_ready.set();
    _tmp_event1_l1_ready.wait();
    for (int j = 0; j < 10; j += 1) {
        _tmp_event2_l1_valid.wait();
        if (m == 0) {
            GM2L1_ND2NZ(l1q.get(cnt), x[K*m], 128, K, K, 128);
        }
        _tmp_event2_l1_ready.set();
        _tmp_event2_l1_ready.wait();
        for (int k = 0; k < 20; k += 1) {
            _tmp_event1_l0_valid.wait();
            L0NZ2ZZ(l0a.get(cnt), l1q.get(cnt)[1024], 64, K, 128, K);
            L0NZ2NZ(l0b.get(cnt), l1k.get(cnt), 64, K, 128, K);
            _tmp_event1_l0_ready.set();
            _tmp_event1_l0_ready.wait();
            _tmp_event1_fix_valid.wait();
            MMAD(l0c.get(cnt), l0a.get(cnt), l0b.get(cnt), 64, K, 64, true);
            _tmp_event1_fix_ready.set();
            _tmp_event1_fix_ready.wait();
            _tmp_event1_l0_valid.set();
            _tmp_event1_l0_valid.wait();
            L0NZ2ZZ(l0a.get(cnt), l1q.get(cnt)[1024], 64, K, 128, K);
            L0NZ2NZ(l0b.get(cnt), l1k.get(cnt), 64, K, 128, K);
            _tmp_event1_l0_ready.set();
            _tmp_event1_l0_ready.wait();
            _tmp_event1_fix_valid.wait();
            MMAD(l0c.get(cnt), l0a.get(cnt), l0b.get(cnt), 64, K, 64, true);
            _tmp_event1_fix_ready.set();
            _tmp_event1_fix_ready.wait();
            _tmp_event1_l0_valid.set();
            L0C2GM_NZ2ND(y, l0c.get(cnt), 1, N, N, 128);
            _tmp_event1_fix_valid.set();
        }
        _tmp_event2_l1_valid.set();
    }
    _tmp_event1_l1_valid.set();
}
// end auto sync
for (int m = 0; m < m2; m += 128) {
    // start auto sync
    _tmp_sevent1_l1_valid.wait();
    GM2L1_ND2NZ(l1q.get(cnt), x[K*m], 128, K, K, 128);
    GM2L1_ND2NZ(l1k.get(cnt), x[K*m], 128, K, K, 128);
    GM2L1_ND2NZ(l1v, x[K*m], 128, K, K, 128);
    _tmp_sevent1_l1_ready.set();
    _tmp_sevent1_l1_ready.wait();
    int j = 0;
    _tmp_event1_l0_valid.wait();
    for (; j < 10; j += 1) {
        L0NZ2ZZ(l0a.get(cnt), l1q.get(cnt)[1024], 64, K, 128, K);
        L0NZ2NZ(l0b.get(cnt), l1k.get(cnt), 64, K, 128, K);
    }
    _tmp_event1_l0_ready.set();
    _tmp_event1_l0_ready.wait();
    _tmp_sevent1_l1_valid.set();
    _tmp_event1_fix_valid.wait();
    MMAD(l0c.get(cnt), l0a.get(cnt), l0b.get(cnt), 64, K, 64, true);
    _tmp_event1_fix_ready.set();
    _tmp_event1_fix_ready.wait();
    _tmp_event1_l0_valid.set();
    L0C2GM_NZ2ND(y, l0c.get(cnt), 1, N, N, 128);
    _tmp_event1_fix_valid.set();
    // end auto sync
}
for (int m = 0; m < m2; m += 128) {
    // start auto sync
    _tmp_sevent1_l1_valid.wait();
    GM2L1_ND2NZ(l1q.get(cnt), x[K*m], 128, K, K, 128);
    GM2L1_ND2NZ(l1k.get(cnt), x[K*m], 128, K, K, 128);
    GM2L1_ND2NZ(l1v, x[K*m], 128, K, K, 128);
    _tmp_sevent1_l1_ready.set();
    _tmp_sevent1_l1_ready.wait();
    int j = 0;
    _tmp_event1_l0_valid.wait();
    for (; j < 10; j += 1) {
        L0NZ2ZZ(l0a.get(cnt), l1q.get(cnt)[1024], 64, K, 128, K);
        L0NZ2NZ(l0b.get(cnt), l1k.get(cnt), 64, K, 128, K);
    }
    _tmp_event1_l0_ready.set();
    _tmp_event1_l0_ready.wait();
    _tmp_sevent1_l1_valid.set();
    _tmp_sevent1_fix_valid.wait();
    MMAD(l0c.get(cnt), l0a.get(cnt), l0b.get(cnt), 64, K, 64, true);
    _tmp_sevent1_fix_ready.set();
    _tmp_sevent1_fix_ready.wait();
    _tmp_event1_l0_valid.set();
    L0C2GM_NZ2ND(y, l0c2, 1, N, N, 128);
    _tmp_sevent1_fix_valid.set();
    // end auto sync
}
for (int m = 0; m < m2; m += 128) {
    // start auto sync
    // end auto sync
}
