#include "base/tensor.h"
#include "base/util.h"
#include <iostream>
#include <gtest/gtest.h>
#include <vector>

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

using namespace std;
using namespace mllm;
using namespace mllm::base;

void print_ts3(Tensor ts, base::Device device = base::Device::CPU)
{
    ts.toDevice(Device::CPU);
    for (size_t i = 0; i < ts.shape(1); i++)
    {
        for (size_t h = 0; h < ts.shape(0); h++)
        {
            for (size_t j = 0; j < ts.shape(2); j++)
            {
                std::cout << *ts[{h, i, j}] << ' ';
            }
            std::cout << "   ";
        }
        cout << endl;
    }
    cout << endl;
    ts.toDevice(device);
}

TEST(TensorCatTest, Demo)
{
    Tensor k_cache({4, 3, 4}, Device::CPU, true, nullptr);
    Tensor k_proj({4, 1, 4}, Device::CPU, true, nullptr);

    for (size_t i = 0; i < k_cache.size(); i++)
        *k_cache[i] = i;
    for (size_t i = 0; i < k_proj.size(); i++)
        *k_proj[i] = i + k_cache.size();

    cout << "k_cache: \n";
    print_ts3(k_cache);
    cout << "k_proj: \n";
    print_ts3(k_proj);

    Tensor k_cat = k_cache.clone();
    k_cat.cat(k_proj, 1);
    cout << "k_cat:\n";
    print_ts3(k_cat);

    size_t tot = k_cat.logic_size();

    for (size_t i = 0; i < tot; i++)
    {
        auto idx = k_cat.index(i);
        if (idx[1] < k_cache.shape(1))
        {
            EXPECT_EQ(*k_cat[i], *k_cache[idx]);
        }
        else
        {
            idx[1] -= k_cache.shape(1);
            EXPECT_EQ(*k_cat[i], *k_proj[idx]);
        }
    }
}

TEST(TensorCatTestCuda, DemoCuda)
{
    Tensor k_cache({4, 3, 4}, Device::CPU, true, nullptr);
    Tensor k_proj({4, 1, 4}, Device::CPU, true, nullptr);

    for (size_t i = 0; i < k_cache.size(); i++)
        *k_cache[i] = i;
    for (size_t i = 0; i < k_proj.size(); i++)
        *k_proj[i] = i + k_cache.size();

    cout << "k_cache: \n";
    print_ts3(k_cache, Device::CUDA);
    cout << "k_proj: \n";
    print_ts3(k_proj, Device::CUDA);

    Tensor k_cat = k_cache.clone();
    k_cat.cat(k_proj, 1);
    cout << "k_cat:\n";
    print_ts3(k_cat, Device::CUDA);

    size_t tot = k_cat.logic_size();
    k_cache.toDevice(Device::CPU);
    k_cat.toDevice(Device::CPU);
    k_proj.toDevice(Device::CPU);
    for (size_t i = 0; i < tot; i++)
    {
        auto idx = k_cat.index(i);
        if (idx[1] < k_cache.shape(1))
        {
            EXPECT_EQ(*k_cat[i], *k_cache[idx]);
        }
        else
        {
            idx[1] -= k_cache.shape(1);
            EXPECT_EQ(*k_cat[i], *k_proj[idx]);
        }
    }
}