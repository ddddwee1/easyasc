from easyasc.a5 import * 


BLK = 64

@vf()
def addmic(x: Tensor, y: Tensor, n_rows: Var, u8_buf: Tensor):
    r_x = Reg(x.dtype)
    r_y = Reg(y.dtype)
    r_out = Reg(y.dtype)
    mask = MaskReg(x.dtype)
    t = Var(1.0, DT.half)
    cfg_f32 = CastConfig()

    ub_to_reg(r_x, x, mask=mask)
    ub_to_reg(r_y, y, mask=mask)
    add(r_out, r_x, r_y, mask=mask)
    adds(r_out, r_out, n_rows, mask=mask)
    add(r_out, r_x, r_y)
    muls(r_x, r_x, t)
    reg_to_ub(y, r_out, mask=mask)

    # micro stub coverage
    r1 = Reg(x.dtype)
    r2 = Reg(x.dtype)
    r3 = Reg(x.dtype)
    r4 = Reg(x.dtype)
    mask2 = MaskReg(x.dtype)
    mask3 = MaskReg(x.dtype)
    idx_u16 = Reg(DT.uint16)
    cnt_u32 = Var(0, DT.uint32)
    r_u8 = Reg(DT.uint8)
    r_u8_b = Reg(DT.uint8)
    mask_u8 = MaskReg(DT.uint8)
    r_f = Reg(DT.float)
    ub_u8 = u8_buf

    arange(idx_u16, 0)
    dup(r2, n_rows, mask=mask)
    cast(r_f, r_x, cfg_f32, mask)
    compare(mask2, r_x, r_y, CompareMode.LT, mask=mask)
    compare(mask3, r_x, 0.0, CompareMode.GT, mask=mask)
    select(r3, r_x, r_y, mask=mask2)

    vmax(r1, r_x, r_y, mask=mask)
    vmin(r2, r_x, r_y, mask=mask)
    sub(r3, r_x, r_y, mask=mask)
    mul(r4, r_x, r_y, mask=mask)
    div(r1, r_x, r_y, mask=mask)
    vand(r2, r_x, r_y, mask=mask)
    vor(r3, r_x, r_y, mask=mask)
    vxor(r4, r_x, r_y, mask=mask)
    prelu(r1, r_x, r_y, mask=mask)

    exp(r1, r_x, mask=mask)
    abs(r2, r_x, mask=mask)
    relu(r3, r_x, mask=mask)
    sqrt(r4, r_x, mask=mask)
    ln(r1, r_x, mask=mask)
    log(r2, r_x, mask=mask)
    log2(r3, r_x, mask=mask)
    log10(r4, r_x, mask=mask)
    neg(r1, r_x, mask=mask)
    vnot(r2, r_x, mask=mask)
    vcopy(r3, r_x, mask=mask)

    vmaxs(r1, r_x, 1.0, mask=mask)
    vmins(r2, r_x, n_rows, mask=mask)
    adds(r3, r_x, 2.0, mask=mask)
    muls(r4, r_x, 2.0, mask=mask)
    lrelu(r1, r_x, 0.1, mask=mask)
    shiftls(r2, r_x, 1, mask=mask)
    shiftrs(r3, r_x, 1, mask=mask)
    axpy(r4, r_x, 2.0, mask=mask)

    cmax(r1, r_x, mask=mask)
    cgmax(r2, r_x, mask=mask)
    cmin(r3, r_x, mask=mask)
    cgmin(r4, r_x, mask=mask)
    cadd(r1, r_x, mask=mask)
    cgadd(r2, r_x, mask=mask)
    cpadd(r3, r_x, mask=mask)

    interleave(r1, r2, r_x, r_y)
    deinterleave(r3, r4, r1, r2)

    mask_not(mask2, mask, mask=mask3)
    mask_and(mask3, mask, mask2, mask=mask)
    mask_or(mask2, mask2, mask3, mask=mask)
    mask_xor(mask3, mask, mask2, mask=mask)
    mask_mov(mask2, mask3, mask=mask)
    mask_interleave(mask2, mask3, mask, mask2)
    mask_deinterleave(mask2, mask3, mask, mask2)
    mask_sel(mask2, mask, mask3, mask=mask)
    mask_pack(mask2, mask3, low_part=True)
    mask_unpack(mask3, mask2, low_part=False)
    move_mask_spr(mask2)
    update_mask(mask2, cnt_u32)

    ub_to_reg_continuous(r1, x, LoadDist.SINGLE_VALUE)
    reg_to_ub_continuous(y, r1, mask, StoreDist.NORMAL)
    reg_to_ub_downsample(y, r1, mask=mask)
    reg_to_ub_pack4(ub_u8, r_u8, mask=mask_u8)
    reg_to_ub_single(y, r1, mask=mask)
    ub_to_reg_single(r1, x)
    ub_to_reg_upsample(r_u8, ub_u8)
    ub_to_reg_downsample(r_u8, ub_u8)
    ub_to_reg_unpack(r_u8, ub_u8)
    ub_to_reg_unpack4(r_u8, ub_u8)
    ub_to_reg_brcb(r1, x)
    ub_to_reg_gather(r1, x, idx_u16, mask=mask)
    reg_to_ub_scatter(y, r1, idx_u16, mask=mask)
    gather(r1, r_x, idx_u16)
    gather_mask(r1, r_x, mask=mask)

    ub_to_reg(r_u8_b, ub_u8, mask=mask_u8)
    reg_to_ub(ub_u8, r_u8_b, mask=mask_u8)


@kernel()
def cubefunc(x: GMTensor, y: GMTensor, z: GMTensor, M: Var, N: Var, K: Var):
    z.bind_cv_mutex(0)

    l1x = DBuff(DT.half, [BLK, K], Position.L1)
    l1y = DBuff(DT.half, [N, K], Position.L1)
    l0c = DBuff(DT.float, [BLK, N], Position.L0C)
    xub = DBuff(DT.half, [BLK, N], Position.UB)
    outub = DBuff(DT.half, [BLK, N], Position.UB)
    u8ub = Tensor(DT.uint8, [1, BLK], Position.UB)

    cnt = Var(0)

    m_per_core = CeilDiv(M, GetCubeNum())
    m1 = Var(m_per_core * GetCubeIdx())
    m2 = Min(m1 + m_per_core, M)

    with auto_sync():
        for m in range(m1, m2, BLK):
            l1x[cnt] <<= x[m:m+BLK, :]
            l1y[cnt] <<= y[:, :]
            
            matmul(l0c[cnt], l1x[cnt], l1y[cnt])
            z.lock()
            z[m:m+BLK, :] <<= l0c[cnt]
            z.ready()
            z.wait()
            xub[cnt] <<= z[m:m+BLK, :]
            z.free()
            
            if GetSubBlockIdx()==0:
                addmic(xub[0], outub[1], m_per_core, u8ub)
                z[m:m+BLK, :] <<= outub[cnt]

    return z 


if __name__ == "__main__":
    import torch 

    out_dir = "test_cust_op"

    M = 64*20 
    N = 64 
    K = 128 
    x = torch.randn(M, K).half()
    y = torch.randn(N, K).half()
    z = torch.randn(M, N).half()

    op = OpExec(cubefunc, out_dir, gen_only=True)
    op(x, y, z, M, N, K)

