#pragma once 
#include <cassert>

#define CHECK_RET(x)                                                                        \
    do {                                                                                    \
        auto __ret = x;                                                                 \
        if (__ret != ACL_SUCCESS) {                                                      \
            std::cerr << __FILE__ << ":" << __LINE__ << " Error:" << __ret << std::endl; \
        }                                                                                   \
    } while (0);


#define PREPARE_OP() \
    uint64_t workspaceSize = 0; \
    aclOpExecutor* executor = nullptr; \
    void* workspaceAddr = nullptr; 


#define EXECOP(opname, stream, ...) \
    CHECK_RET(opname##GetWorkspaceSize(__VA_ARGS__, &workspaceSize, &executor));\
    if (workspaceSize>0)  CHECK_RET(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST)); \
    CHECK_RET(opname(workspaceAddr, workspaceSize, executor, stream)); \
    if (workspaceSize>0)  CHECK_RET(aclrtFree(workspaceAddr));


#define INT4 int8_t, aclDataType::ACL_INT4
#define INT8 int8_t, aclDataType::ACL_INT8
#define INT16 int16_t, aclDataType::ACL_INT16
#define INT32 int32_t, aclDataType::ACL_INT32
#define INT64 int64_t, aclDataType::ACL_INT64
#define BF16 int16_t, aclDataType::ACL_BF16
#define FP16 int16_t, aclDataType::ACL_FLOAT16
#define FP32 float, aclDataType::ACL_FLOAT
#define BOOL char, aclDataType::ACL_BOOL
#define HIF8 uint8_t, aclDataType::ACL_HIFLOAT8
#define UINT8 uint8_t, aclDataType::ACL_UINT8
#define UINT16 uint16_t, aclDataType::ACL_UINT16
#define UINT32 uint32_t, aclDataType::ACL_UINT32
#define UINT64 uint64_t, aclDataType::ACL_UINT64

#define assertm(exp, msg) assert(((void)msg, exp))

#define PROFILE_ALL ACL_PROF_ACL_API|ACL_PROF_TASK_TIME|ACL_PROF_AICORE_METRICS|ACL_PROF_HCCL_TRACE
