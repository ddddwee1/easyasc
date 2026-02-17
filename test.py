from easyasc.a5 import *
from easyasc.utils.regop import RegOP


BLK = 64

@vf()
def addmic(x: Tensor, y: Tensor, n_rows: Var, u8_buf: Tensor):
    r_x = Reg(x.dtype)
    r_y = Reg(y.dtype)
    r_out = Reg(y.dtype)
    mask = MaskReg(x.dtype)
    t = Var(1.0, DT.half)
    cfg_f32 = CastConfig()

    r1 = Reg(x.dtype)
    r2 = Reg(x.dtype)
    r3 = Reg(x.dtype)
    r4 = Reg(x.dtype)
    r5 = Reg(x.dtype)
    r6 = Reg(x.dtype)
    mask2 = MaskReg(x.dtype)
    mask3 = MaskReg(x.dtype)
    mask4 = MaskReg(x.dtype)
    mask5 = MaskReg(x.dtype)
    idx_u16 = Reg(DT.uint16)
    cnt_u32 = Var(0, DT.uint32)
    r_u8 = Reg(DT.uint8)
    r_u8_b = Reg(DT.uint8)
    mask_u8 = MaskReg(DT.uint8)
    r_f = Reg(DT.float)
    ub_u8 = u8_buf

    # Cast
    r1 <<= ((r2 + r3) * (r4 - r1)).exp().sqrt()
    r_f <<= x[10]

    r_x <<= x[8]
    r_y <<= y[8]
    if n_rows > 64:
        r_out <<= r_x + r_y
    elif n_rows > 0:
        r_out <<= r_x - r_y
    else:
        r_out <<= r_x * r_y
    y[8] <<= r_out

    for i in range(0, n_rows):
        r1 <<= x[i]
        y[i] <<= r1

    x[12] <<= r_f


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
