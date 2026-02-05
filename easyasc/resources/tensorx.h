#pragma once
#include "acl/acl.h"
#include "cust_op_list.h"
#include "macros.h"
#include <iostream>
#include <fstream>
#include <cstdio>

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <string>
#include <vector>


template <typename T, aclDataType aclT>
class TensorX{
protected: 
    void* m_host_ptr = nullptr; 
    void* m_device_ptr = nullptr;
    std::vector<int64_t> m_shape;
    std::vector<int64_t> m_strides;
    int64_t m_numel = 0;
    int64_t m_data_size = 0;

public: 
    TensorX(){}
    void _init(std::vector<int64_t> shape){
        m_shape = shape;
        m_strides = shape;
        m_numel = 1;
        if (shape.size() > 0){
            m_strides[shape.size()-1] = 1;
            for (int i=shape.size()-2; i>=0; --i){
                m_strides[i] = m_strides[i+1] * shape[i+1];
            }

            for (int i=0; i<shape.size(); ++i){
                m_numel *= shape[i];
            }
        }
        m_data_size = m_numel * sizeof(T);
    }

    TensorX(std::vector<int64_t> shape){
        _init(shape);
        m_host_ptr = nullptr;
        m_device_ptr = nullptr;
    }

    void fillHost(T val){
        if (m_host_ptr==nullptr){
            initHost();
        }
        T* host_ptr = (T*) m_host_ptr;
        if constexpr (aclT==aclDataType::ACL_INT4){
            for (int i=0; i<(m_numel+1)/2; ++i){
                host_ptr[i] = (val << 4) | (val & 0xf) ;
            }
        }else{
            for (int i=0; i<m_numel; ++i){
                host_ptr[i] = val;
            }
        }
    }

    void fillHostWithBinFile(const std::string &filePath){
        std::ifstream ifs(filePath);
        ifs.seekg(0, std::ios::beg);
        ifs.read((char*)m_host_ptr, m_numel*sizeof(T));
        ifs.close();
    }

    void saveHostToBinFile(const std::string &filePath){
        assertm(m_host_ptr!=nullptr, "Should call copyToHost before saving to bin file");

        int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);

        auto writeSize = write(fd, (void*) m_host_ptr, m_data_size);
        (void) close(fd);
    }

    void initHost(){
        if (m_host_ptr==nullptr){
            if constexpr (aclT==aclDataType::ACL_INT4){
                m_host_ptr = (void*) new T[(m_numel+1)/2];
            }else{
                m_host_ptr = (void*) new T[m_numel];
            }
        }
    }

    void initDevice(){
        if (m_device_ptr==nullptr){
            aclrtMalloc(&m_device_ptr, m_data_size, ACL_MEM_MALLOC_HUGE_FIRST);
        }
    }

    void initAll(){
        initHost();
        initDevice();
    }

    void freeHost(){
        if (m_host_ptr!=nullptr){
            delete (T*) m_host_ptr;
            m_host_ptr = nullptr;
        }
    }

    void freeDevice(){
        if (m_device_ptr!=nullptr){
            aclrtFree(m_device_ptr);
            m_device_ptr = nullptr;
        }
    }

    void freeAll(){
        freeHost();
        freeDevice();
    }

    aclTensor* toAclTensor(){
        if (m_device_ptr==nullptr){
            printf("[ERROR] Input tensor is not on device. Please use initDevice or copyToDevice before forward\n");
            throw std::invalid_argument("Tensor not on device");
        }
        return aclCreateTensor(m_shape.data(), m_shape.size(), aclT, m_strides.data(), 0, aclFormat::ACL_FORMAT_ND, m_shape.data(), m_shape.size(), m_device_ptr);
    }

    void copyToHost(){
        if (m_device_ptr!=nullptr){
            if (m_host_ptr==nullptr){
                initHost();
            }
            CHECK_RET(aclrtMemcpy(m_host_ptr, m_data_size, m_device_ptr, m_data_size, ACL_MEMCPY_DEVICE_TO_HOST));
        }
    }

    void copyToDevice(){
        if (m_host_ptr!=nullptr){
            if (m_device_ptr==nullptr){
                initDevice();
            }
            CHECK_RET(aclrtMemcpy(m_device_ptr, m_data_size, m_host_ptr, m_data_size, ACL_MEMCPY_HOST_TO_DEVICE));
        }
    }

};


