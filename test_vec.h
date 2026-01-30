DBuff<float, TPosition::VECCALC> xub;
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
for (int m = 0; m < m2; m += 128) {
    // start auto sync
    cnt = cnt + 1;
    // end auto sync
}
for (int m = 0; m < m2; m += 128) {
    // start auto sync
    cnt = cnt + 1;
    // end auto sync
}
for (int m = 0; m < m2; m += 128) {
    // start auto sync
    cnt = cnt + 1;
    // end auto sync
}
// start auto sync
// end auto sync
// start auto sync
// end auto sync
// start auto sync
// end auto sync
// start auto sync
// end auto sync
// start auto sync
// end auto sync
for (int m = 0; m < m2; m += 128) {
    // start auto sync
    // end auto sync
}
for (int m = 0; m < m2; m += 128) {
    // start auto sync
    // end auto sync
}
for (int m = 0; m < m2; m += 128) {
    WAIT_CUBE(0, PIPE_S);
    Add<float, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 64), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
    // start auto sync
    Sub<float, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 64), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
    Mul<float, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 64), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
    Div<float, false>(xub.get(cnt), xub.get(cnt1), xub.get(cnt2), MASK_PLACEHOLDER, CeilDiv(128*K, 64), {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8});
    // end auto sync
}
