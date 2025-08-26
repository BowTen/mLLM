#include "kernel/kernel.h"
#include <gtest/gtest.h>
#include "model/qwen3_rotary_embedding.h"
#include "model/qwen3.h"

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

using namespace std;
using namespace mllm;
using namespace mllm::base;

// class CPUGenRoPE : public ::testing::Test
// {

// protected:
//     Tensor Q;
//     Tensor cos;
//     Tensor sin;
//     void SetUp() override
//     {
//         google::InitGoogleLogging("CPURoPE");
//         FLAGS_logtostderr = true;
//         VLOG(DEBUG) << "Setting up CPURoPE test environment";
//         std::vector<size_t> input_shape({2, 2, 4});
//         std::vector<size_t> weight_shape({2, 4});
//         vector<float> Q_data({1.0f, 2.0f, 3.0f, 4.0f,
//                               5.0f, 6.0f, 7.0f, 8.0f,

//                               1.0f, 2.0f, 3.0f, 4.0f,
//                               5.0f, 6.0f, 7.0f, 8.0f});

//         vector<float> cos_data({std::cos(0.1f), std::cos(0.2f), std::cos(0.1f), std::cos(0.2f),
//                                 std::cos(0.3f), std::cos(0.4f), std::cos(0.3f), std::cos(0.4f)});
//         vector<float> sin_data({std::sin(0.1f), std::sin(0.2f), std::sin(0.1f), std::sin(0.2f),
//                                 std::sin(0.3f), std::sin(0.4f), std::sin(0.3f), std::sin(0.4f)});

//         Q = Tensor(Q_data.data(), input_shape, true);
//         cos = Tensor(cos_data.data(), weight_shape, true);
//         sin = Tensor(sin_data.data(), weight_shape, true);
//     }

//     void TearDown() override
//     {
//         google::ShutdownGoogleLogging();
//     }
// };

// TEST_F(CPURoPE, PrintCPURoPEResult)
// {
//     cout << "Q:\n";
//     for (size_t i = 0; i < Q.shape(0); ++i)
//     {
//         for (size_t j = 0; j < Q.shape(1); ++j)
//         {
//             for (size_t k = 0; k < Q.shape(2); ++k)
//             {
//                 cout << *Q[{i, j, k}] << ' ';
//             }
//             cout << endl;
//         }
//         cout << endl;
//     }
//     cout << endl;
//     cout << "cos:\n";
//     for (size_t i = 0; i < cos.shape(0); ++i)
//     {
//         for (size_t j = 0; j < cos.shape(1); ++j)
//         {
//             cout << *cos[{i, j}] << ' ';
//         }
//         cout << endl;
//     }
//     cout << endl;
//     cout << "sin:\n";
//     for (size_t i = 0; i < sin.shape(0); ++i)
//     {
//         for (size_t j = 0; j < sin.shape(1); ++j)
//         {
//             cout << *sin[{i, j}] << ' ';
//         }
//         cout << endl;
//     }
//     cout << endl;

//     VLOG(DEBUG) << "Running RoPE kernel on CPU";
//     kernel::get_rope_kernel(Device::CPU)(&Q, &cos, &sin, &Q, nullptr);
//     cout << "Q:\n";
//     for (size_t i = 0; i < Q.shape(0); ++i)
//     {
//         for (size_t j = 0; j < Q.shape(1); ++j)
//         {
//             for (size_t k = 0; k < Q.shape(2); ++k)
//             {
//                 cout << *Q[{i, j, k}] << ' ';
//             }
//             cout << endl;
//         }
//         cout << endl;
//     }
//     cout << endl;
// }

// TEST_F(CPURoPE, PrintCUDARoPEResult)
// {
//     cout << "Q:\n";
//     for (size_t i = 0; i < Q.shape(0); ++i)
//     {
//         for (size_t j = 0; j < Q.shape(1); ++j)
//         {
//             for (size_t k = 0; k < Q.shape(2); ++k)
//             {
//                 cout << *Q[{i, j, k}] << ' ';
//             }
//             cout << endl;
//         }
//         cout << endl;
//     }
//     cout << endl;
//     cout << "cos:\n";
//     for (size_t i = 0; i < cos.shape(0); ++i)
//     {
//         for (size_t j = 0; j < cos.shape(1); ++j)
//         {
//             cout << *cos[{i, j}] << ' ';
//         }
//         cout << endl;
//     }
//     cout << endl;
//     cout << "sin:\n";
//     for (size_t i = 0; i < sin.shape(0); ++i)
//     {
//         for (size_t j = 0; j < sin.shape(1); ++j)
//         {
//             cout << *sin[{i, j}] << ' ';
//         }
//         cout << endl;
//     }
//     cout << endl;

//     VLOG(DEBUG) << "Running RoPE kernel on CUDA";
//     Q.toDevice(Device::CUDA);
//     cos.toDevice(Device::CUDA);
//     sin.toDevice(Device::CUDA);
//     kernel::get_rope_kernel(Device::CUDA)(&Q, &cos, &sin, &Q, nullptr);
//     Q.toDevice(Device::CPU);
//     cout << "Q:\n";
//     for (size_t i = 0; i < Q.shape(0); ++i)
//     {
//         for (size_t j = 0; j < Q.shape(1); ++j)
//         {
//             for (size_t k = 0; k < Q.shape(2); ++k)
//             {
//                 cout << *Q[{i, j, k}] << ' ';
//             }
//             cout << endl;
//         }
//         cout << endl;
//     }
//     cout << endl;
// }

class RoPECheck : public ::testing::Test
{
protected:
    model::Qwen3 qwen3_cuda;

    float check_eps = 1e-6;
    model::Qwen3RotaryEmbedding rope_cpu;
    model::Qwen3RotaryEmbedding rope_cuda;

    std::mt19937 rnd = std::mt19937(std::random_device{}());
    std::uniform_real_distribution<> urd = std::uniform_real_distribution<>(-1.0, 1.0);

    RoPECheck() : qwen3_cuda(model::Qwen3::from_pretrained("/home/hznuojai/ai_infra/MiniLLM/resources/Qwen/Qwen3-0.6B", base::Device::CUDA, 1.0)),
                  rope_cpu(qwen3_cuda.config(), base::Device::CPU, static_cast<cudaStream_t>(nullptr)),
                  rope_cuda(qwen3_cuda.config(), base::Device::CUDA, static_cast<cudaStream_t>(nullptr)) {}

    void SetUp() override
    {
        google::InitGoogleLogging("RoPECheck");
        FLAGS_logtostderr = true;
        VLOG(DEBUG) << "Setting up RoPECheck test environment";
    }

    void TearDown() override
    {
        google::ShutdownGoogleLogging();
    }
};

TEST_F(RoPECheck, CPUvsCUDA)
{
    Tensor cos_cpu = Tensor({2, 128}, base::Device::CPU);
    Tensor sin_cpu = Tensor({2, 128}, base::Device::CPU);
    Tensor cos_cuda = Tensor({2, 128}, base::Device::CUDA);
    Tensor sin_cuda = Tensor({2, 128}, base::Device::CUDA);

    PosEmb pos_emb_cpu(&cos_cpu, &sin_cpu);
    PosEmb pos_emb_cuda(&cos_cuda, &sin_cuda);

    rope_cpu.forward(0, 2, pos_emb_cpu);
    rope_cuda.forward(0, 2, pos_emb_cuda);
    cudaDeviceSynchronize();

    cos_cuda.toDevice(base::Device::CPU);
    sin_cuda.toDevice(base::Device::CPU);

    size_t total_size = cos_cpu.size();
    for (size_t i = 0; i < total_size; i++)
    {
        EXPECT_NEAR(*cos_cpu[i], *cos_cuda[i], check_eps);
        EXPECT_NEAR(*sin_cpu[i], *sin_cuda[i], check_eps);
    }
}