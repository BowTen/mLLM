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

void print_ts3(Tensor ts)
{
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
}

TEST(TensorCatTest, Demo)
{
    Tensor k_cache({4, 3, 4}, Device::CPU, true);
    Tensor k_proj({4, 1, 4}, Device::CPU, true);
    for (size_t i = 0; i < k_cache.size(); i++)
        *k_cache[i] = base::get_random_float();
    for (size_t i = 0; i < k_proj.size(); i++)
        *k_proj[i] = base::get_random_float();

    cout << "k_cache: \n";
    print_ts3(k_cache);
    cout << "k_proj: \n";
    print_ts3(k_proj);

    k_cache.cat(k_proj, 1);
    cout << "new k_cache:\n";
    print_ts3(k_cache);
}