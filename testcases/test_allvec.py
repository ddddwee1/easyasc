import easyasc as ea 
from easyasc import * 


BLK = 128


@ea.kernel()
def cubefunc(x: GMTensor, y: GMTensor, z: GMTensor, M: Var, N: Var, K: Var):
    l1q = DBuff(DT.half, [BLK, K], Position.L1)
    l1k = DBuff(DT.half, [BLK, K], Position.L1)
    l1v = Tensor(DT.half, [BLK, K], Position.L1)
    l0a = DBuff(DT.half, [BLK, K], Position.L0A)
    l0b = DBuff(DT.half, [BLK, K], Position.L0B)
    l0c = DBuff(DT.float, [BLK, K], Position.L0C)
    xub = DBuff(DT.float, [BLK, K], Position.UB)
    xub_int = DBuff(DT.int, [BLK, K], Position.UB)
    xub_half = Tensor(DT.half, [BLK, K], Position.UB)
    xub_mask = Tensor(DT.uint8, [BLK, K], Position.UB)
    offset = Tensor(DT.uint32, [BLK, K], Position.UB)
    mask_high = Var(0, dtype=DT.uint64)
    mask_low = Var(0, dtype=DT.uint64)

    in_ready = DEvent(Pipe.MTE2, Pipe.MTE1)
    in_valid = DEvent(Pipe.MTE1, Pipe.MTE2)
    l1_ready = DEvent(Pipe.MTE1, Pipe.M)
    l1_valid = DEvent(Pipe.M, Pipe.MTE1)
    out_ready = DEvent(Pipe.M, Pipe.FIX)
    out_valid = DEvent(Pipe.FIX, Pipe.M)

    cnt = Var(0)

    m_per_core = CeilDiv(M, GetCubeNum())
    m1 = Var(m_per_core * GetCubeIdx())
    m2 = Min(m1 + m_per_core, M)

    for m in range(m1, m2, BLK):
        wait_vec()
        in_valid.wait()
        l1q[cnt] <<= x[m:m + BLK, 0:K]
        in_ready.set()
        l1_ready.wait()
        l0a[cnt] <<= l1q[cnt][BLK//2:BLK, 0:K]
        l0b[cnt] <<= l1k[cnt][0:BLK//2, 0:K]
        in_valid.set()

        mmad(l0c[cnt], l0a[cnt], l0b[cnt])

        with atomic_min():
            z[m:m+BLK, :] <<= l0c[cnt]

        cnt += 1 

        cube_ready()
        allcube_ready()
        allcube_wait()

    for m in range(m1, m2, BLK):
        wait_cube()
        add(xub[cnt], xub[cnt], xub[cnt])
        add(xub[cnt][:32, :64], xub[cnt][:32, :64], xub[cnt][:32, :64])
        sub(xub[cnt], xub[cnt], xub[cnt])
        mul(xub[cnt], xub[cnt], xub[cnt])
        div(xub[cnt], xub[cnt], xub[cnt])
        vmax(xub[cnt], xub[cnt], xub[cnt])
        vmin(xub[cnt], xub[cnt], xub[cnt])
        muladddst(xub[cnt], xub[cnt], xub[cnt])
        sub(xub[cnt][:32, :64], xub[cnt][:32, :64], xub[cnt][:32, :64])
        vand(xub_int[cnt], xub_int[cnt], xub_int[cnt])
        vor(xub_int[cnt], xub_int[cnt], xub_int[cnt])
        exp(xub[cnt], xub[cnt])
        ln(xub[cnt], xub[cnt])
        abs(xub[cnt], xub[cnt])
        rec(xub[cnt], xub[cnt])
        sqrt(xub[cnt], xub[cnt])
        rsqrt(xub[cnt], xub[cnt])
        relu(xub[cnt], xub[cnt])
        vnot(xub_int[cnt], xub_int[cnt])
        adds(xub[cnt], xub[cnt], 1.5)
        muls(xub[cnt], xub[cnt], 2)
        vmaxs(xub[cnt], xub[cnt], 0)
        vmins(xub[cnt], xub[cnt], 0)
        lrelu(xub[cnt], xub[cnt], 0.1)
        axpy(xub[cnt][:32, :64], xub[cnt][:32, :64], 0.5)
        dup(xub[cnt], 0.25)
        brcb(xub[cnt], xub[cnt])
        gather(xub[cnt], xub[cnt], offset)
        scatter(xub[cnt], xub[cnt], offset)
        gm_to_ub_pad(xub_half, x, 1, 16, 0, 0)
        gm_to_ub_pad(xub_half, x)
        xub_half <<= x
        ub_to_gm_pad(y, xub_half, 1, 16, 0, 0)
        ub_to_gm_pad(y, xub_half)
        y <<= xub_half
        with atomic_add():
            ub_to_gm_pad(y, xub_half, 1, 16, 0, 0)
        with atomic_max():
            ub_to_gm_pad(y, xub_half, 1, 16, 0, 0)
        ub_to_ub(xub_half, xub_half, 1, 16, 0, 0)
        cmax(xub[cnt], xub[cnt])
        cgmax(xub[cnt], xub[cnt])
        cmin(xub[cnt], xub[cnt])
        cgmin(xub[cnt], xub[cnt])
        cadd(xub[cnt], xub[cnt])
        cgadd(xub[cnt], xub[cnt])
        cpadd(xub[cnt], xub[cnt])
        set_mask(mask_high, mask_low)
        reset_mask()
        cast(xub_half, xub[cnt])
        cast(xub[cnt], xub_half, round_mode=RoundMode.FLOOR)
        compare(xub_mask, xub[cnt], xub[cnt], CompareMode.EQ)
        compare_scalar(xub_mask, xub[cnt], 0, CompareMode.GT)
        set_cmpmask(xub_mask)
        _xub_half_cast = reinterpret(xub[cnt], DT.half)
        bar_m()
        bar_v()
        bar_mte3()
        bar_mte2()
        bar_mte1()
        bar_fix()
        bar_all()
        select(xub[cnt], xub_mask, xub[cnt], xub[cnt])
        select(xub[cnt], xub_mask, xub[cnt], 0)
        exp(xub[cnt][:32, :64], xub[cnt][:32, :64])
        vec_ready()
        allvec_ready()
        allvec_wait()


if __name__ == "__main__":
    M = Var(64)
    N = Var(64)
    K = Var(64)
    x= GMTensor(DT.half, [M, K])
    y= GMTensor(DT.half, [K, N])
    z= GMTensor(DT.half, [M, N])
    cubefunc(x, y, z, M, N, K)
    # cubefunc.print_instructions()
    cubefunc.dump_asc("test.cpp")
