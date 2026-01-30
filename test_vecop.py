import easyasc as ea 
from easyasc import * 


BLK = 128


@ea.kernel
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
    ubin_ready = DEvent(Pipe.MTE2, Pipe.V)
    ubin_valid = DEvent(Pipe.V, Pipe.MTE2)

    cnt = Var(0)
    cnt1 = Var(0)
    cnt2 = Var(0)
    ubcnt = Var(0)
    val = Var(1.0)

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

        cnt += 1 

        cube_ready()
        allcube_ready()
        allcube_wait()


    for m in range(m1, m2, BLK):
        wait_cube()
        ubin_valid.wait()
        ubin_ready.set()
        xub[cnt] <<= xub[cnt1] + xub[cnt2]
        xub[cnt] <<= xub[cnt1] - xub[cnt2]
        xub[cnt] <<= xub[cnt1] * xub[cnt2]
        xub[cnt] <<= xub[cnt1] / xub[cnt2]
        xub[cnt] <<= xub[cnt1] + 1.0
        xub[cnt] <<= xub[cnt1] - 1.0
        xub[cnt] <<= xub[cnt1] * 1.0
        xub[cnt] <<= xub[cnt1].exp()
        xub[cnt] <<= xub[cnt1].ln()
        xub[cnt] <<= xub[cnt1].abs()
        xub[cnt] <<= xub[cnt1].rec()
        xub[cnt] <<= xub[cnt1].sqrt()
        xub[cnt] <<= xub[cnt1].rsqrt()
        xub[cnt] <<= xub[cnt1].vnot()
        xub[cnt] <<= xub[cnt1].relu()

        xub[cnt] <<= xub[cnt1].cmax()
        xub[cnt] <<= xub[cnt1].cgmax()
        xub[cnt] <<= xub[cnt1].cmin()
        xub[cnt] <<= xub[cnt1].cgmin()
        xub[cnt] <<= xub[cnt1].cadd()
        xub[cnt] <<= xub[cnt1].cgadd()
        xub[cnt] <<= xub[cnt1].cpadd()

        xub[cnt] <<= xub[cnt1].cast()

        xub[cnt] <<= xub[cnt1] & xub[cnt2]
        xub[cnt] <<= xub[cnt1] | xub[cnt2]
        xub[cnt] <<= maximum(xub[cnt1], xub[cnt2])
        xub[cnt] <<= minimum(xub[cnt1], xub[cnt2])

        xub[cnt] <<= maximum(xub[cnt1], 1.0)
        xub[cnt] <<= minimum(xub[cnt1], 1.0)
        xub[cnt] <<= maximum(xub[cnt1], val)
        xub[cnt] <<= minimum(xub[cnt1], val)
        


if __name__ == "__main__":
    M = Var(64)
    N = Var(64)
    K = Var(64)
    x= GMTensor(DT.half, [M, K])
    y= GMTensor(DT.half, [K, N])
    z= GMTensor(DT.half, [M, N])
    cubefunc(x, y, z, M, N, K)
    # cubefunc.print_instructions()
    cubefunc.dump_asc("test")
