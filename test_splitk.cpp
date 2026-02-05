#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "tensorutils.h"
#include "test_splitk_cube.h"
#include "test_splitk_vec.h"


extern "C" __global__ __aicore__ void test_splitk(GM_ADDR x, GM_ADDR y_t, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(tiling_data, tiling);
    PipeBarrier<PIPE_ALL>();
    int M = tiling_data.M;
    int N = tiling_data.N;
    int K = tiling_data.K;
    int splitk = tiling_data.splitk;
    if ASCEND_IS_AIC{
        cubefunc_splitk_cube(x, y_t, z, workspace, M, N, K, splitk);
    }
    if ASCEND_IS_AIV{
        cubefunc_splitk_vec(x, y_t, z, workspace, M, N, K, splitk);
    }

}
