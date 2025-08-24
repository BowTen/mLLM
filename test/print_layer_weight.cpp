#include "model/qwen3.h"
#include <gtest/gtest.h>

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

using namespace std;
using namespace mllm;
using namespace mllm::base;
using namespace mllm::model;

int main(int argc, char **argv)
{
    testing::InitGoogleTest();
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;

    // 运行所有测试
    int result = RUN_ALL_TESTS();

    // 清理 Google Logging
    google::ShutdownGoogleLogging();

    return result;
}

// class Qwen3WeightPrint : public ::testing::Test
// {
// protected:
//     Qwen3 model;

//     Qwen3WeightPrint() : model(Qwen3::from_pretrained("/home/hznuojai/ai_infra/MiniLLM/resources/Qwen/Qwen3-0.6B", base::Device::CPU, 5.0f))
//     {
//         VLOG(DEBUG) << "Set up Qwen3WeightPrint";
//     }

//     void TearDown() override
//     {
//         VLOG(DEBUG) << "Tearing down Qwen3WeightPrint";
//     }
// };

// string shape_str(vector<size_t> shape)
// {
//     string s = "[";
//     for (auto dim : shape)
//     {
//         s += to_string(dim) + ",";
//     }
//     s.back() = ']';
//     return s;
// }

// TEST_F(Qwen3WeightPrint, Demo)
// {
//     LOG(INFO) << "Print Qwen3 layer weights:";
//     auto wlayers = model.weighted_layers();
//     for (auto layer : wlayers)
//     {
//         auto name = layer->name();
//         size_t pos = name.find("model.layers.");
//         if (pos != std::string::npos)
//         {
//             std::string layer_num_str = name.substr(pos + 13, 2);
//             if (layer_num_str != "27")
//             {
//                 continue;
//             }
//         }

//         auto shape = layer->weight().shape();
//         cout << "\n=== 参数名称: " << name << " ===" << endl;
//         cout << "参数形状: " << shape_str(shape) << endl;

//         auto weight = layer->weight();
//         if (*min_element(shape.begin(), shape.end()) == 1)
//         {
//             cout << "前10个参数值:" << endl;
//             for (size_t i = 0; i < 10; ++i)
//             {
//                 cout << *weight[i] << ' ';
//             }
//             cout << endl;
//             cout << "后10个参数值:" << endl;
//             for (size_t i = weight.size() - 10; i < weight.size(); ++i)
//             {
//                 cout << *weight[i] << ' ';
//             }
//             cout << endl;
//         }
//         else
//         {
//             cout << "左上角 4x4 矩阵:" << endl;
//             for (size_t i = 0; i < 4; ++i)
//             {
//                 for (size_t j = 0; j < 4; ++j)
//                 {
//                     cout << *weight[{i, j}] << ' ';
//                 }
//                 cout << endl;
//             }

//             size_t rows = shape[0];
//             size_t cols = shape[1];
//             cout << "右下角 4x4 矩阵:" << endl;
//             for (size_t i = rows - 4; i < rows; ++i)
//             {
//                 for (size_t j = cols - 4; j < cols; ++j)
//                 {
//                     cout << *weight[{i, j}] << ' ';
//                 }
//                 cout << endl;
//             }
//         }
//         cout << "-----------------------------------------" << endl;
//     }
// }

class Qwen3WeightPrintCuda : public ::testing::Test
{
protected:
    Qwen3 model;

    Qwen3WeightPrintCuda() : model(Qwen3::from_pretrained("/home/hznuojai/ai_infra/MiniLLM/resources/Qwen/Qwen3-0.6B", base::Device::CUDA, 5.0f))
    {
        VLOG(DEBUG) << "Set up Qwen3WeightPrintCuda";
    }

    void TearDown() override
    {
        VLOG(DEBUG) << "Tearing down Qwen3WeightPrintCuda";
    }
};

string shape_str(vector<size_t> shape)
{
    string s = "[";
    for (auto dim : shape)
    {
        s += to_string(dim) + ",";
    }
    s.back() = ']';
    return s;
}

TEST_F(Qwen3WeightPrintCuda, Demo)
{
    LOG(INFO) << "Print Qwen3 layer weights:";
    auto wlayers = model.weighted_layers();
    for (auto layer : wlayers)
    {
        auto name = layer->name();
        size_t pos = name.find("model.layers.");
        if (pos != std::string::npos)
        {
            std::string layer_num_str = name.substr(pos + 13, 2);
            if (layer_num_str != "27")
            {
                continue;
            }
        }

        auto shape = layer->weight().shape();
        cout << "\n=== 参数名称: " << name << " ===" << endl;
        cout << "参数形状: " << shape_str(shape) << endl;

        auto weight = layer->weight();
        weight.toDevice(base::Device::CPU);
        if (*min_element(shape.begin(), shape.end()) == 1)
        {
            cout << "前10个参数值:" << endl;
            for (size_t i = 0; i < 10; ++i)
            {
                cout << *weight[i] << ' ';
            }
            cout << endl;
            cout << "后10个参数值:" << endl;
            for (size_t i = weight.size() - 10; i < weight.size(); ++i)
            {
                cout << *weight[i] << ' ';
            }
            cout << endl;
        }
        else
        {
            cout << "左上角 4x4 矩阵:" << endl;
            for (size_t i = 0; i < 4; ++i)
            {
                for (size_t j = 0; j < 4; ++j)
                {
                    cout << *weight[{i, j}] << ' ';
                }
                cout << endl;
            }

            size_t rows = shape[0];
            size_t cols = shape[1];
            cout << "右下角 4x4 矩阵:" << endl;
            for (size_t i = rows - 4; i < rows; ++i)
            {
                for (size_t j = cols - 4; j < cols; ++j)
                {
                    cout << *weight[{i, j}] << ' ';
                }
                cout << endl;
            }
        }
        cout << "-----------------------------------------" << endl;
        weight.toDevice(base::Device::CUDA);
    }
}