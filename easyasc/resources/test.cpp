#include <iostream>
#include <cstdio>
#include <cstring>
#include <vector>
#include <fstream>
#include <sys/stat.h>
#include "tensorx.h"
#include "acl/acl.h"
#include "acl/acl_prof.h"
#include "macros.h"


int main(int argc, char **argv)
{
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = aclInit(nullptr);
    ret = aclrtSetDevice(deviceId);
    ret = aclrtCreateContext(&context, deviceId);
    ret = aclrtSetCurrentContext(context);
    ret = aclrtCreateStream(&stream);
    
    // const char *aclProfPath = "./";
    // aclprofInit(aclProfPath, strlen(aclProfPath));
    // uint32_t devices[1];
    // devices[0] = 0;
    // aclprofConfig *config = aclprofCreateConfig((uint32_t*)devices, 1, ACL_AICORE_ARITHMETIC_UTILIZATION, nullptr, PROFILE_ALL);
    // const char *memFreq = "15";
    // aclprofSetConfig(ACL_PROF_SYS_HARDWARE_MEM_FREQ, memFreq, strlen(memFreq));
    // aclprofStart(config);
    
    printf("--> Initializing tensors...\n");

    
    // printf("--> Finalizing profiling...\n");
    // aclprofStop(config);
    // aclprofDestroyConfig(config);
    // aclprofFinalize();
    
    aclFinalize();
}
