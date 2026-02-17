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

    # Arithmetic (RegOP via operators)
    r_out <<= r_x + r_y
    r_out <<= r_x - r_y
    r_out <<= r_x * r_y
    r_out <<= r_x / r_y
    r_out <<= r_x + 5.0
    r_out <<= r_x - 3.0
    r_out <<= r_x * 1.2
    r_out <<= r_x / 2.0
    r_out <<= r_x * t

    # Binary ops
    r1 <<= r_x.vmax(r_y)
    r2 <<= r_x.vmin(r_y)
    r3 <<= r_x.vand(r_y)
    r4 <<= r_x.vor(r_y)
    r5 <<= r_x.vxor(r_y)
    r6 <<= r_x.prelu(r_y)

    # Unary ops
    r1 <<= r_x.exp()
    r2 <<= r_x.abs()
    r3 <<= r_x.relu()
    r4 <<= r_x.sqrt()
    r5 <<= r_x.ln()
    r6 <<= r_x.log()
    r1 <<= r_x.log2()
    r2 <<= r_x.log10()
    r3 <<= r_x.neg()
    r4 <<= r_x.vnot()
    r5 <<= r_x.vcopy()

    # Scalar ops
    r1 <<= r_x.vmaxs(1.1)
    r2 <<= r_x.vmins(n_rows)
    r5 <<= r_x.lrelu(0.1)
    r6 <<= r_x.shiftls(1)
    r1 <<= r_x.shiftrs(1)
    r2 <<= r_x.axpy(2.0)

    # Group ops
    r1 <<= r_x.cmax()
    r2 <<= r_x.cgmax()
    r3 <<= r_x.cmin()
    r4 <<= r_x.cgmin()
    r5 <<= r_x.cadd()
    r6 <<= r_x.cgadd()
    r1 <<= r_x.cpadd()

    # Dup + arange
    r2 <<= 3.0 
    r3 <<= r_x.dup()
    idx_u16.arange(1, True)

    # Cast
    r_f <<= r_x.astype(DT.float, cfg_f32)

    # Compare + select
    mask2 <<= r_x < r_y
    mask3 <<= r_x > 0.0
    r4 <<= mask2.select(r_x, r_y)

    # Mask ops
    mask4 <<= ~mask2
    mask2 <<= mask2 & mask3
    mask3 <<= mask2 | mask4
    mask4 <<= mask2 ^ mask3
    mask2 <<= mask2.mov(mask3)
    mask3 <<= mask2.sel(mask3, mask4)
    mask2 <<= mask2.pack(True)
    mask3 <<= mask2.unpack(False)

    # Datamove variants (RegOP + lshift)
    y <<= r_x
    y <<= r_x.downsample()
    y <<= r_x.single_value()
    ub_u8 <<= r_u8.pack4()

    r_u8_b <<= u8_buf
    r1 <<= x.single()
    r1 <<= x.upsample()


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
