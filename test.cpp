GlobalTensor<half> x;
x.SetGlobalBuffer((__gm__ half*) x_);
GlobalTensor<half> y;
y.SetGlobalBuffer((__gm__ half*) y_);
GlobalTensor<half> z;
z.SetGlobalBuffer((__gm__ half*) z_);
DBuff<half, TPosition::A1> l1q;
DBuff<half, TPosition::A1> l1k;
LocalTensor<half> l1v;
DBuff<half, TPosition::A2> l0a;
DBuff<half, TPosition::B2> l0b;
DBuff<float, TPosition::CO1> l0c;
DBuff<float, TPosition::VECCALC> xub;
DBuff<int, TPosition::VECCALC> xub_int;
LocalTensor<half> xub_half;
LocalTensor<uint8_t> xub_mask;
LocalTensor<uint32_t> offset;
uint64_t mask_high = 0;
uint64_t mask_low = 0;
DEvent<PIPE_MTE2, PIPE_MTE1> in_ready;
DEvent<PIPE_MTE1, PIPE_MTE2> in_valid;
DEvent<PIPE_MTE1, PIPE_M> l1_ready;
DEvent<PIPE_M, PIPE_MTE1> l1_valid;
DEvent<PIPE_M, PIPE_FIX> out_ready;
DEvent<PIPE_FIX, PIPE_M> out_valid;
int cnt = 0;
int m_per_core = CeilDiv(M, GetBlockNum());
int m1 = m_per_core*get_block_idx();
int m2 = Min(m1 + m_per_core, M);
for (int m = 0; m < m2; m += 128) {
    WAIT_VEC(0, PIPE_S);
    in_valid.wait();
    GM2L1_ND2NZ(l1q.get(cnt), x[K*m], 128, K, K, 128);
    in_ready.set();
    l1_ready.wait();
    L0NZ2ZZ(l0a.get(cnt), l1q.get(cnt)[1024], 64, K, 128, K);
    L0NZ2NZ(l0b.get(cnt), l1k.get(cnt), 64, K, 128, K);
    in_valid.set();
    MMAD(l0c.get(cnt), l0a.get(cnt), l0b.get(cnt), 64, K, 64, true);
    if ((m == 0) && (K > 0)) {
        GM2L1_ND2NZ(l1q.get(cnt), x[K*m], 128, K, K, 128);
    } else if ((m > 1) || (K == 0)) {
        GM2L1_ND2NZ(l1k.get(cnt), y, K, N, N, 128);
    } else {
        GM2L1_ND2NZ(l1q.get(cnt), x[K*(m + 128)], 128, K, K, 128);
    }
    if (!(m == 0)) {
        GM2L1_ND2NZ(l1q.get(cnt), x[K*m], 128, K, K, 128);
    }
    if ((!(m == 0)) && ((K > 0) || (N > 0))) {
        GM2L1_ND2NZ(l1k.get(cnt), y, K, N, N, 128);
    }
    cnt = cnt + 1;
    CUBE_READY(0, PIPE_FIX);
    ALLCUBE_READY(0, PIPE_FIX);
    ALLCUBE_WAIT(0, PIPE_S);
}
