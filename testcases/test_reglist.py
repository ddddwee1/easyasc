from easyasc.a5 import *
from easyasc.parser.asc import translate


BLK = 128
LANES = 4


@vf()
def addmic(x: Tensor, y: Tensor, out: Tensor):
    f32_reglist = RegList(DT.float, LANES)
    src = RegList(x.dtype, LANES)
    rhs = RegList(x.dtype, LANES)
    dst = RegList(x.dtype, LANES)
    scale = Var(3.0, x.dtype)


    f32_reglist <<= x[10]

    src <<= x
    rhs <<= y

    # binary: reglist <op> reglist
    dst <<= src + rhs
    dst <<= src - rhs
    dst <<= src * rhs
    dst <<= src / rhs
    dst <<= src.vmax(rhs)
    dst <<= src.vmin(rhs)
    dst <<= src.vand(rhs)
    dst <<= src.vor(rhs)
    dst <<= src.vxor(rhs)
    dst <<= src.prelu(rhs)

    # binary: reglist <op> reg
    base = Reg(x.dtype)
    base <<= x[0]
    dst <<= src + base
    dst <<= src - base
    dst <<= src * base
    dst <<= src / base

    # binary: reglist <op> scalar
    dst <<= src + 2.0
    dst <<= src - 2.0
    dst <<= src * 2.0
    dst <<= src / 2.0

    # unary
    dst <<= src.exp()
    dst <<= src.abs()
    dst <<= src.sqrt()
    dst <<= src.relu()
    dst <<= src.ln()
    dst <<= src.log()
    dst <<= src.log2()
    dst <<= src.log10()
    dst <<= src.neg()
    dst <<= src.vnot()
    dst <<= src.vcopy()

    # unary + scalar
    dst <<= src.shiftls(1)
    dst <<= src.shiftrs(1)
    dst <<= src.axpy(scale)
    dst <<= src.lrelu(0.1)
    dst <<= src.vmaxs(1.0)
    dst <<= src.vmins(scale)

    # reduce
    r_add = Reg(x.dtype)
    r_max = Reg(x.dtype)
    r_min = Reg(x.dtype)
    r_add <<= src.cadd()
    r_max <<= src.cmax()
    r_min <<= src.cmin()

    out[0] <<= r_add
    out[1] <<= r_max
    out[2] <<= r_min

    out <<= src 
    out[10] <<= f32_reglist


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
                addmic(xub[0], outub[1], outub[0])
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
