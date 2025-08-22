// #include "kernel/kernel.h"
// #include <gtest/gtest.h>

// #define GLOG_USE_GLOG_EXPORT
// #include <glog/logging.h>

// using namespace std;
// using namespace mllm;
// using namespace mllm::base;

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

// class RoPECheck : public ::testing::Test
// {
// protected:
//     Tensor Q;
//     Tensor cos;
//     Tensor sin;

//     float check_eps = 1e-6;

//     std::mt19937 rnd = std::mt19937(std::random_device{}());
//     std::uniform_real_distribution<> urd = std::uniform_real_distribution<>(-1.0, 1.0);

//     void SetUp() override
//     {
//         google::InitGoogleLogging("RoPECheck");
//         FLAGS_logtostderr = true;
//         VLOG(DEBUG) << "Setting up RoPECheck test environment";
//         std::vector<size_t> input_shape({8, 32, 128});
//         std::vector<size_t> weight_shape({32, 128});

//         Q = Tensor(input_shape);
//         cos = Tensor(weight_shape);
//         sin = Tensor(weight_shape);

//         for (size_t i = 0; i < Q.size(); i++)
//             *Q[i] = urd(rnd);
//         for (size_t i = 0; i < cos.size(); i++)
//         {
//             *cos[i] = urd(rnd);
//             *sin[i] = urd(rnd);
//         }
//     }

//     void TearDown() override
//     {
//         google::ShutdownGoogleLogging();
//     }
// };

// TEST_F(RoPECheck, CPUvsCUDA)
// {
//     cout << "CPUvsCUDA\n";
//     cout << "Q: ";
//     for (int i = 0; i < 5; i++)
//         cout << *Q[i] << ' ';
//     cout << "...\n";

//     cout << "cos: ";
//     for (int i = 0; i < 5; i++)
//         cout << *cos[i] << ' ';
//     cout << "...\n";

//     cout << "sin: ";
//     for (int i = 0; i < 5; i++)
//         cout << *sin[i] << ' ';
//     cout << "...\n";

//     VLOG(DEBUG) << "clone data to cuda";
//     Tensor Qcuda = Q.clone();
//     Tensor coscuda = cos.clone();
//     Tensor sincuda = sin.clone();
//     Qcuda.toDevice(Device::CUDA);
//     coscuda.toDevice(Device::CUDA);
//     sincuda.toDevice(Device::CUDA);

//     VLOG(DEBUG) << "Running RoPE kernel on CPU";
//     kernel::get_rope_kernel(Device::CPU)(&Q, &cos, &sin, &Q, nullptr);
//     VLOG(DEBUG) << "Running RoPE kernel on CUDA";
//     kernel::get_rope_kernel(Device::CUDA)(&Qcuda, &coscuda, &sincuda, &Qcuda, nullptr);
//     Qcuda.toDevice(Device::CPU);

//     cout << "Qcpu  Qcuda\n";
//     int diff = 0;
//     for (size_t i = 0; i < Q.size(); i++)
//     {
//         if (i < 5)
//             cout << *Q[i] << ' ' << *Qcuda[i] << '\n';
//         diff += std::fabs(*Q[i] - *Qcuda[i]) > check_eps;
//     }
//     EXPECT_EQ(diff, 0) << "Mismatch found " << diff << " between CPU and CUDA results";
// }