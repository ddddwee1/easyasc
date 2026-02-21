from easyasc.a5 import *


BLK = 128
LANES = 4

@kernel()
def cubefunc(x: GMTensor, y: GMTensor, z: GMTensor, M: Var, N: Var, K: Var):
    z.bind_cv_mutex(0)

    l1x = DBuff(DT.half, [BLK, K], Position.L1)
    l1y = DBuff(DT.half, [N, K], Position.L1)
    l0c = DBuff(DT.float, [BLK, N], Position.L0C)

    cnt = Var(0)

    m_per_core = CeilDiv(M, GetCubeNum())
    m1 = Var(m_per_core * GetCubeIdx())
    m2 = Min(m1 + m_per_core, M)

    with auto_sync():
        for m in range(m1, m2, BLK):
            l1x[cnt] <<= x[m:m+BLK, :]
            if GetCubeIdx()==0:
                sim_print(l1x[cnt], pipe=Pipe.MTE2)
            # l1y[cnt] <<= y[:, :]
            
            # matmul(l0c[cnt], l1x[cnt], l1y[cnt])
            # z[m:m+BLK, :] <<= l0c[cnt]

    return z 


if __name__ == "__main__":
    import torch 

    out_dir = "test_cust_op"

    M = 64*64 
    N = 64 
    K = 128 
    x = torch.randn(M, K).half()
    for k in unroll(K):
        x[:,k] = k
    y = torch.randn(N, K).half()
    z = torch.randn(M, N).half()

    op = OpExec(cubefunc, out_dir, simulator=True)
    op(x, y, z, M, N, K)
