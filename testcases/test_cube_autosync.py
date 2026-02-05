import easyasc as ea 
from easyasc import * 


BLK = 128


@func()
@auto_sync()
def subset_vec(xub: DBuff, cnt: Var, cnt1: Var, cnt2: Var):
    xub[cnt] <<= xub[cnt1] - xub[cnt2]
    xub[cnt] <<= xub[cnt1] * xub[cnt2]
    xub[cnt] <<= xub[cnt1] / xub[cnt2]


@kernel()
def cubefunc(x: GMTensor, y: GMTensor, z: GMTensor, M: Var, N: Var, K: Var):
    l1q = DBuff(DT.half, [BLK, K], Position.L1)
    l1k = DBuff(DT.half, [BLK, K], Position.L1)
    l1v = Tensor(DT.half, [BLK, K], Position.L1)
    l0a = DBuff(DT.half, [BLK, K], Position.L0A)
    l0b = DBuff(DT.half, [BLK, K], Position.L0B)
    l0c = DBuff(DT.float, [BLK, K], Position.L0C)
    l0c2 = Tensor(DT.float, [BLK, K], Position.L0C)
    xub = DBuff(DT.float, [BLK, K], Position.UB)

    cnt = Var(0)
    cnt1 = Var(0)
    cnt2 = Var(0)
    ubcnt = Var(0)
    val = Var(1.0)

    m_per_core = CeilDiv(M, GetCubeNum())
    m1 = Var(m_per_core * GetCubeIdx())
    m2 = Min(m1 + m_per_core, M)

    # test case 1
    for m in range(m1, m2, BLK):
        with auto_sync():
            l1q[cnt] <<= x[m:m + BLK, 0:K]
            l0a[cnt] <<= l1q[cnt][BLK//2:BLK, 0:K]
            l0b[cnt] <<= l1k[cnt][0:BLK//2, 0:K]

            mmad(l0c[cnt], l0a[cnt], l0b[cnt])

            y[:1, :] <<= l0c[cnt]

            cnt += 1 

    # test case 2
    for m in range(m1, m2, BLK):
        with auto_sync():
            l1q[cnt] <<= x[m:m + BLK, 0:K]
            for j in range(10):
                l0a[cnt] <<= l1q[cnt][BLK//2:BLK, 0:K]
                l0b[cnt] <<= l1k[cnt][0:BLK//2, 0:K]
                mmad(l0c[cnt], l0a[cnt], l0b[cnt])
            y[:1, :] <<= l0c[cnt]

            cnt += 1 

    # test case 3
    for m in range(m1, m2, BLK):
        with auto_sync():
            l1q[cnt] <<= x[m:m + BLK, 0:K]
            l1k[cnt] <<= x[m:m + BLK, 0:K]
            for j in range(10):
                l0a[cnt] <<= l1q[cnt][BLK//2:BLK, 0:K]
                l0b[cnt] <<= l1k[cnt][0:BLK//2, 0:K]
            mmad(l0c[cnt], l0a[cnt], l0b[cnt])
            y[:1, :] <<= l0c[cnt]
            
            cnt += 1 

    # test case 4
    for m in range(m1, m2, BLK):
        with auto_sync():
            l1q[cnt] <<= x[m:m + BLK, 0:K]

            for j in range(10):
                l1k[cnt] <<= x[m:m + BLK, 0:K]
                l0a[cnt] <<= l1q[cnt][BLK//2:BLK, 0:K]
                l0b[cnt] <<= l1k[cnt][0:BLK//2, 0:K]

            mmad(l0c[cnt], l0a[cnt], l0b[cnt])
            y[:1, :] <<= l0c[cnt]
            
            cnt += 1 

    # test case 5
    with auto_sync():
        l1q[cnt] <<= x[m:m + BLK, 0:K]
        for m in range(m1, m2, BLK):
            l1q[cnt] <<= x[m:m + BLK, 0:K]
            for j in range(10):
                l1q[cnt] <<= x[m:m + BLK, 0:K]
                for k in range(20):
                    l0a[cnt] <<= l1q[cnt][BLK//2:BLK, 0:K]
                    l0b[cnt] <<= l1k[cnt][0:BLK//2, 0:K]
                    mmad(l0c[cnt], l0a[cnt], l0b[cnt])
                y[:1, :] <<= l0c[cnt]
        
    # test case 6
    with auto_sync():
        for m in range(m1, m2, BLK):
            l1q[cnt] <<= x[m:m + BLK, 0:K]
            for j in range(10):
                if m==0:
                    l1q[cnt] <<= x[m:m + BLK, 0:K]
                for k in range(20):
                    l0a[cnt] <<= l1q[cnt][BLK//2:BLK, 0:K]
                    l0b[cnt] <<= l1k[cnt][0:BLK//2, 0:K]
                    if cnt==0:
                        mmad(l0c[cnt], l0a[cnt], l0b[cnt], is_init=True)
                    else:
                        mmad(l0c[cnt], l0a[cnt], l0b[cnt], is_init=False)
                    if k==1:
                        y[:1, :] <<= l0c[cnt]

    # test case 7
    with auto_sync():
        for m in range(m1, m2, BLK):
            l1q[cnt] <<= x[m:m + BLK, 0:K]
            for j in range(10):
                if m==0:
                    l1q[cnt] <<= x[m:m + BLK, 0:K]
                for k in range(20):
                    l0a[cnt] <<= l1q[cnt][BLK//2:BLK, 0:K]
                    l0b[cnt] <<= l1k[cnt][0:BLK//2, 0:K]
                    mmad(l0c[cnt], l0a[cnt], l0b[cnt])
                    l0a[cnt] <<= l1q[cnt][BLK//2:BLK, 0:K]
                    l0b[cnt] <<= l1k[cnt][0:BLK//2, 0:K]
                    mmad(l0c[cnt], l0a[cnt], l0b[cnt])

    # test case 8
    with auto_sync():
        for m in range(m1, m2, BLK):
            l1q[cnt] <<= x[m:m + BLK, 0:K]
            for j in range(10):
                if m==0:
                    l1q[cnt] <<= x[m:m + BLK, 0:K]
                for k in range(20):
                    l0a[cnt] <<= l1q[cnt][BLK//2:BLK, 0:K]
                    l0b[cnt] <<= l1k[cnt][0:BLK//2, 0:K]
                    mmad(l0c[cnt], l0a[cnt], l0b[cnt])
                    y[:1, :] <<= l0c[cnt]
                    l0a[cnt] <<= l1q[cnt][BLK//2:BLK, 0:K]
                    l0b[cnt] <<= l1k[cnt][0:BLK//2, 0:K]
                    mmad(l0c[cnt], l0a[cnt], l0b[cnt])
                    y[:1, :] <<= l0c[cnt]

    # test case 9
    with auto_sync():
        for m in range(m1, m2, BLK):
            l1q[cnt] <<= x[m:m + BLK, 0:K]
            for j in range(10):
                if m==0:
                    l1q[cnt] <<= x[m:m + BLK, 0:K]
                for k in range(20):
                    l0a[cnt] <<= l1q[cnt][BLK//2:BLK, 0:K]
                    l0b[cnt] <<= l1k[cnt][0:BLK//2, 0:K]
                    mmad(l0c[cnt], l0a[cnt], l0b[cnt])
                    l0a[cnt] <<= l1q[cnt][BLK//2:BLK, 0:K]
                    l0b[cnt] <<= l1k[cnt][0:BLK//2, 0:K]
                    mmad(l0c[cnt], l0a[cnt], l0b[cnt])
                    y[:1, :] <<= l0c[cnt]

    # test case 10
    for m in range(m1, m2, BLK):
        with auto_sync():
            l1q[cnt] <<= x[m:m + BLK, 0:K]
            l1k[cnt] <<= x[m:m + BLK, 0:K]
            l1v <<= x[m:m + BLK, 0:K]
            for j in range(10):
                l0a[cnt] <<= l1q[cnt][BLK//2:BLK, 0:K]
                l0b[cnt] <<= l1k[cnt][0:BLK//2, 0:K]
            mmad(l0c[cnt], l0a[cnt], l0b[cnt])
            y[:1, :] <<= l0c[cnt]

    # test case 11
    for m in range(m1, m2, BLK):
        with auto_sync():
            l1q[cnt] <<= x[m:m + BLK, 0:K]
            l1k[cnt] <<= x[m:m + BLK, 0:K]
            l1v <<= x[m:m + BLK, 0:K]
            for j in range(10):
                l0a[cnt] <<= l1q[cnt][BLK//2:BLK, 0:K]
                l0b[cnt] <<= l1k[cnt][0:BLK//2, 0:K]
            mmad(l0c[cnt], l0a[cnt], l0b[cnt])
            y[:1, :] <<= l0c2


    for m in range(m1, m2, BLK):
        wait_cube()
        xub[cnt] <<= xub[cnt1] + xub[cnt2]
        subset_vec(xub, cnt, cnt1, cnt2)


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
