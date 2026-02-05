import easyasc as ea 
from easyasc import * 
from easyasc.shortcuts.matmul import matmul
import os


BLK = 128


@func()
def subset_vec(xub: DBuff, cnt: Var, cnt1: Var, cnt2: Var):
    xub[cnt] <<= xub[cnt1] - xub[cnt2]
    xub[cnt] <<= xub[cnt1] * xub[cnt2]
    xub[cnt] <<= xub[cnt1] / xub[cnt2]


@kernel()
def cubefunc(x: GMTensor, y: GMTensor, z: GMTensor, M: Var, N: Var, K: Var):
    y.bind_cv_mutex(0)
    vv = split_workspace(DT.half, [M, N])
    l1q = DBuff(DT.half, [BLK, K], Position.L1)
    l1k = DBuff(DT.half, [BLK, K], Position.L1)
    l1v = Tensor(DT.half, [BLK, K], Position.L1)
    l0a = DBuff(DT.half, [BLK, K], Position.L0A)
    l0b = DBuff(DT.half, [BLK, K], Position.L0B)
    l0c = DBuff(DT.float, [BLK, K], Position.L0C)
    l0c2 = Tensor(DT.float, [BLK, K], Position.L0C)
    xub = DBuff(DT.half, [BLK, K], Position.UB)
    reset_cache()
    xubs = Tensor(DT.half, [BLK, K], Position.UB)

    cnt = Var(0)
    cnt1 = Var(0)
    cnt2 = Var(0)
    ubcnt = Var(0)
    val = Var(1.0)

    m_per_core = CeilDiv(M, GetCubeNum())
    m1 = Var(m_per_core * GetCubeIdx())
    m2 = Min(m1 + m_per_core, M)

    for m in range(m1, m2, BLK):
        with auto_sync():
            l1q[cnt] <<= x[m:m + BLK, 0:K]
            matmul(
                l0c[cnt],
                l1q[cnt][BLK // 2 : BLK, 0:K],
                l1k[cnt][0 : BLK // 2, 0:K],
            )
            l1v <<= l0c[cnt]

            y.lock()
            y[:1, :] <<= l0c[cnt]
            cnt += 1 
            y.ready()


    with auto_sync():
        for m in range(m1, m2, BLK):
            y.wait()
            xubs <<= y[m:m + BLK, 0:K]
            y.free()
            for i in range(10):
                xub[cnt] <<= x[m:m + BLK, 0:K]
                xub[cnt] <<= xub[cnt1] + xub[cnt2]
            subset_vec(xub, cnt, cnt1, cnt2)
            vv[m:m + BLK, 0:K] <<= xub[cnt]

    xub[cnt] <<= xub[cnt1] + xub[cnt2]
    return z 


@kernel()
def cubefunc_splitn(x: GMTensor, y_t: GMTensor, z: GMTensor, M: Var, N: Var, K: Var, splitn: Var):
    l1a = Tensor(DT.half, [BLK, K], Position.L1)
    l1b = Tensor(DT.half, [N, K], Position.L1)
    l0c = Tensor(DT.float, [BLK, N], Position.L0C)
    with auto_sync():
        l1a <<= x[0:BLK, 0:K]
        l1b <<= y_t[0:N, 0:K]
        matmul(l0c, l1a, l1b, splitn=splitn)
        z[0:BLK, 0:N] <<= l0c
    return z 


@kernel()
def cubefunc_splitk(x: GMTensor, y_t: GMTensor, z: GMTensor, M: Var, N: Var, K: Var, splitk: Var):
    l1a = Tensor(DT.half, [BLK, K], Position.L1)
    l1b = Tensor(DT.half, [N, K], Position.L1)
    l0c = Tensor(DT.float, [BLK, N], Position.L0C)
    with auto_sync():
        l1a <<= x[0:BLK, 0:K]
        l1b <<= y_t[0:N, 0:K]
        matmul(l0c, l1a, l1b, splitk=splitk, is_init=True)
        matmul(l0c, l1a, l1b, splitk=splitk, is_init=False)
        z[0:BLK, 0:N] <<= l0c
    return z 


if __name__ == "__main__":
    M = Var(64)
    N = Var(64)
    K = Var(64)
    x= GMTensor(DT.half, [M, K])
    y= GMTensor(DT.half, [K, N])
    z= GMTensor(DT.half, [M, N])
    y_t = GMTensor(DT.half, [N, K])
    z_splitn = GMTensor(DT.float, [M, N])
    z_splitk = GMTensor(DT.float, [M, N])
    cubefunc(x, y, z, M, N, K)
    # cubefunc.print_instructions()
    # cubefunc.dump_asc("test")
    cubefunc.dump_kernel('test')
    cubefunc.generate_op_host()
    out_dir = "test_cust_op"
    cann_path = os.environ.get("ASCEND_CANN_PACKAGE_PATH", "/home/ma-user/work/ascend-toolkit/latest")
    cubefunc.generate_op_project(out_dir, cann_path)
    splitn = Var(32)
    splitk = Var(32)
    cubefunc_splitn(x, y_t, z_splitn, M, N, K, splitn)
    cubefunc_splitn.dump_kernel("test_splitn")
    cubefunc_splitk(x, y_t, z_splitk, M, N, K, splitk)
    cubefunc_splitk.dump_kernel("test_splitk")

