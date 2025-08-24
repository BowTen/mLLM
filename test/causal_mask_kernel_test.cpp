#include "base/tensor.h"
#include "base/util.h"
#include <iostream>
#include <gtest/gtest.h>
#include <vector>
#include "kernel/kernel.h"
#include "op/softmax.h"
#include "op/causal_mask.h"

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

using namespace std;
using namespace mllm;
using namespace mllm::base;

class CausalMaskTest : public ::testing::Test
{
protected:
    vector<size_t> shape;
    vector<float> data;
    Tensor a;
    op::Softmax softmax_cpu;
    op::Softmax softmax_cuda;
    op::CausalMask causal_mask_cpu;
    op::CausalMask causal_mask_cuda;

    CausalMaskTest() : shape({2, 4, 4}),
                       data({1.0, 2.0, 3.0, 4.0,
                             5.0, 6.0, 7.0, 8.0,
                             1.0, 2.0, 3.0, 4.0,
                             5.0, 6.0, 7.0, 8.0,

                             1.0, 6.0, 7.0, 4.0,
                             5.0, 2.0, 3.0, 8.0,
                             1.0, 6.0, 7.0, 4.0,
                             5.0, 2.0, 3.0, 8.0}),
                       a(data.data(), shape, true, Device::CPU),
                       softmax_cpu(Device::CPU),
                       softmax_cuda(Device::CUDA),
                       causal_mask_cpu(Device::CPU),
                       causal_mask_cuda(Device::CUDA)
    {
        google::InitGoogleLogging("CausalMaskTest");
        FLAGS_logtostderr = true;
        VLOG(DEBUG) << "Set up CausalMaskTest";
    }

    void TearDown() override
    {
        google::ShutdownGoogleLogging();
    }

    void pt_ts3(Tensor &ts)
    {
        for (size_t i = 0; i < ts.size(); i++)
        {
            cout << *ts[i] << ' ';
            if ((i + 1) % ts.shape(-1) == 0)
                cout << endl;
            if ((i + 1) % (ts.shape(-1) * ts.shape(-2)) == 0)
                cout << endl;
        }
        cout << endl;
    }
};

TEST_F(CausalMaskTest, CPU)
{
    cout << "a:\n";
    pt_ts3(a);

    VLOG(DEBUG) << "Running causal_mask kernel on CPU...";
    causal_mask_cpu.forward(a);
    softmax_cpu.forward(a, a);

    cout << "softmax(causal_mask(a)):\n";
    pt_ts3(a);
}

TEST_F(CausalMaskTest, CUDA)
{
    cout << "a:\n";
    pt_ts3(a);

    VLOG(DEBUG) << "Running causal_mask kernel on CUDA...";
    a.toDevice(Device::CUDA);
    causal_mask_cuda.forward(a);
    softmax_cuda.forward(a, a);
    a.toDevice(Device::CPU);

    cout << "softmax(causal_mask(a)):\n";
    pt_ts3(a);
}

class CausalMaskCheck : public ::testing::Test
{
protected:
    vector<size_t> shape;
    Tensor a;
    float check_eps = 1e-6;
    op::CausalMask causal_mask_cpu;
    op::CausalMask causal_mask_cuda;

    CausalMaskCheck() : shape({8, 1024, 1024}),
                        a(shape),
                        causal_mask_cpu(Device::CPU),
                        causal_mask_cuda(Device::CUDA)
    {
        google::InitGoogleLogging("CausalMaskTest");
        FLAGS_logtostderr = true;
        VLOG(DEBUG) << "Set up CausalMaskTest";

        size_t total_size = a.size();
        for (size_t i = 0; i < total_size; i++)
            *a[i] = base::get_random_float();
    }

    void TearDown() override
    {
        google::ShutdownGoogleLogging();
    }

    void pt_ts3(Tensor &ts)
    {
        for (size_t i = 0; i < ts.size(); i++)
        {
            cout << *ts[i] << ' ';
            if ((i + 1) % ts.shape(-1) == 0)
                cout << endl;
            if ((i + 1) % (ts.shape(-1) * ts.shape(-2)) == 0)
                cout << endl;
        }
        cout << endl;
    }
};

TEST_F(CausalMaskCheck, CPUvsCUDA)
{
    auto b = a.clone();
    b.toDevice(Device::CUDA);

    VLOG(DEBUG) << "Running causal_mask kernel on CPU...";
    causal_mask_cpu.forward(a);
    VLOG(DEBUG) << "Running causal_mask kernel on CUDA...";
    causal_mask_cuda.forward(b);
    b.toDevice(Device::CPU);

    int diff = 0;
    int size = a.size();
    for (int i = 0; i < size; i++)
    {
        diff += std::fabs(*a[i] - *b[i]) > check_eps;
    }
    EXPECT_EQ(diff, 0);
}