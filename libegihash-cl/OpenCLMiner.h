/*
 * OpenCLMiner.h
 *
 *  Created on: Dec 13, 2017
 *      Author: ranjeet
 */
#pragma once

#include "nrgcore/plant.h"
#include "nrgcore/miner.h"
#include <fstream>

#include <cstdint>
#include <mutex>
#include <tuple>

#pragma GCC diagnostic push
#if __GNUC__ >= 6
    #pragma GCC diagnostic ignored "-Wignored-attributes"
#endif
#pragma GCC diagnostic ignored "-Wmissing-braces"
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS true
#define CL_HPP_ENABLE_EXCEPTIONS true
#define CL_HPP_CL_1_2_DEFAULT_BUILD true
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include "CL/cl2.hpp"
#pragma GCC diagnostic pop

// macOS OpenCL fix:
#ifndef CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV
#define CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV       0x4000
#endif

#ifndef CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV
#define CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV       0x4001
#endif

#define OPENCL_PLATFORM_UNKNOWN 0
#define OPENCL_PLATFORM_NVIDIA  1
#define OPENCL_PLATFORM_AMD     2
#define OPENCL_PLATFORM_CLOVER  3


namespace energi {

class OpenCLMiner : public Miner
{
public:
    /* -- default values -- */
    /// Default value of the local work size. Also known as workgroup size.
    static const unsigned c_defaultLocalWorkSize = 192;

    /// Default value of the global work size as a multiplier of the local work size
    static const unsigned c_defaultGlobalWorkSizeMultiplier = 16392;

    OpenCLMiner(const Plant& plant, unsigned index);

    virtual ~OpenCLMiner() override;

    std::tuple<bool, cl::Device, int, int, std::string> getDeviceInfo(int index);

    static unsigned instances()
    {
        return s_numInstances > 0 ? s_numInstances : 1;
    }

    static unsigned getNumDevices();
    static void listDevices();
    static bool configureGPU(unsigned _localWorkSize,
            int _globalWorkSizeMultiplier,
            unsigned _platformId,
            uint64_t _currentBlock,
            unsigned _dagLoadMode,
            unsigned _dagCreateDevice,
            bool noeval,
            bool exit,
            bool _nobinary);
    static void setNumInstances(unsigned _instances);
    static void setDevices(const std::vector<unsigned>& _devices, unsigned _selectedDeviceCount)
    {
        for (unsigned i = 0; i < _selectedDeviceCount; i++) {
            s_devices[i] = _devices[i];
        }
    }


private:
    void trun() override;
    void onSetWork() override {}

    bool init_dag(uint32_t height);

    cl::Context m_context;
    cl::CommandQueue m_queue;
    cl::Kernel m_searchKernel;
    cl::Kernel m_dagKernel;

    std::vector<cl::Buffer> m_dag;
    std::vector<cl::Buffer> m_light;
    std::vector<cl::Buffer> m_header;
    std::vector<cl::Buffer> m_searchBuffer;

    unsigned m_globalWorkSize = 0;
    unsigned m_workgroupSize = 0;
    unsigned m_dagItems = 0;

    static unsigned         s_platformId;
    static unsigned         s_numInstances;
    static bool             s_noBinary;
    static std::vector<int> s_devices;

    /// The local work size for the search
    static unsigned         s_workgroupSize;
    /// The initial global work size for the searches
    static unsigned         s_initialGlobalWorkSize;
};

} /* namespace energi */

