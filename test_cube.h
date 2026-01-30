GlobalTensor<half> x;
x.SetGlobalBuffer((__gm__ half*) x_);
DBuff<half, TPosition::A1> l1q;
DBuff<half, TPosition::A1> l1k;
DBuff<half, TPosition::A2> l0a;
DBuff<half, TPosition::B2> l0b;
DBuff<float, TPosition::CO1> l0c;
int cnt = 0;
int m_per_core = CeilDiv(M, GetBlockNum());
int m1 = m_per_core*get_block_idx();
int m2 = Min(m1 + m_per_core, M);
for (int m = 0; m < m2; m += 128) {
    // start auto sync
    GM2L1_ND2NZ(l1q.get(cnt), x[K*m], 128, K, K, 128);
    L0NZ2ZZ(l0a.get(cnt), l1q.get(cnt)[1024], 64, K, 128, K);
    L0NZ2NZ(l0b.get(cnt), l1k.get(cnt), 64, K, 128, K);
    MMAD(l0c.get(cnt), l0a.get(cnt), l0b.get(cnt), 64, K, 64, true);
    cnt = cnt + 1;
    // end auto sync
    CUBE_READY(0, PIPE_FIX);
    ALLCUBE_READY(0, PIPE_FIX);
    ALLCUBE_WAIT(0, PIPE_S);
}
for (int m = 0; m < m2; m += 128) {
    // start auto sync
    // end auto sync
}
