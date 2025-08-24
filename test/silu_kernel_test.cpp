#include "base/tensor.h"
#include "base/util.h"
#include <iostream>
#include <gtest/gtest.h>
#include <vector>
#include "kernel/kernel.h"
#include "op/silu.h"

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

using namespace std;
using namespace mllm;
using namespace mllm::base;

class SiLUTest : public ::testing::Test
{
protected:
    vector<size_t> shape;
    Tensor a;
    op::SiLU silu_cpu;
    op::SiLU silu_cuda;

    SiLUTest() : shape({2, 4, 4}),
                 a(shape),
                 silu_cpu(Device::CPU),
                 silu_cuda(Device::CUDA)
    {
        google::InitGoogleLogging("SiLUTest");
        FLAGS_logtostderr = true;
        VLOG(DEBUG) << "Set up SiLUTest";

        for (size_t i = 0; i < a.size(); i++)
            *a[i] = base::get_random_float() * 100;
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

TEST_F(SiLUTest, CPU)
{
    cout << "a:\n";
    pt_ts3(a);

    VLOG(DEBUG) << "Running silu kernel on CPU...";
    silu_cpu.forward(a);

    cout << "silu(a):\n";
    pt_ts3(a);
}

TEST_F(SiLUTest, CUDA)
{
    cout << "a:\n";
    pt_ts3(a);

    VLOG(DEBUG) << "Running silu kernel on CUDA...";
    a.toDevice(Device::CUDA);
    silu_cuda.forward(a);
    a.toDevice(Device::CPU);

    cout << "silu(a):\n";
    pt_ts3(a);
}

class SiLUCheck : public ::testing::Test
{
protected:
    vector<size_t> shape;
    Tensor a;
    float check_eps = 1e-6;
    op::SiLU silu_cpu;
    op::SiLU silu_cuda;

    SiLUCheck() : shape({8, 1024, 1024}),
                  a(shape),
                  silu_cpu(Device::CPU),
                  silu_cuda(Device::CUDA)
    {
        google::InitGoogleLogging("SiLUTest");
        FLAGS_logtostderr = true;
        VLOG(DEBUG) << "Set up SiLUTest";

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

TEST_F(SiLUCheck, CPUvsCUDA)
{
    auto b = a.clone();
    b.toDevice(Device::CUDA);

    VLOG(DEBUG) << "Running silu kernel on CPU...";
    silu_cpu.forward(a);
    VLOG(DEBUG) << "Running silu kernel on CUDA...";
    silu_cuda.forward(b);
    b.toDevice(Device::CPU);

    int diff = 0;
    int size = a.size();
    for (int i = 0; i < size; i++)
    {
        diff += std::fabs(*a[i] - *b[i]) > check_eps;
    }
    EXPECT_EQ(diff, 0);
}