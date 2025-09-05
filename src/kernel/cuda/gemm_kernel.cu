#include "kernel/cuda/gemm_kernel.cuh"

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

namespace mllm
{
    namespace kernel
    {
#define BLOCK_TILE_M_L 128
#define BLOCK_TILE_N_L 128
#define BLOCK_TILE_M_S 64
#define BLOCK_TILE_N_S 64
#define BLOCK_TILE_K 8
#define THREAD_TILE_M 8
#define THREAD_TILE_N 8

#define FETCH_FLOAT4(f32) (reinterpret_cast<float4 *>(&(f32))[0])

#define LOAD_TILE_GMEM_TO_REG_A(fg)                                                    \
    if (row_st + TILE_M <= M)                                                          \
    {                                                                                  \
        _Pragma("unroll") for (int i = tid * 4; i < TILE_SIZE_A; i += NUM_THREADS * 4) \
        {                                                                              \
            int row = i / TILE_K;                                                      \
            int col = i % TILE_K;                                                      \
            int ld_id = (i / (NUM_THREADS * 4)) * 4;                                   \
            if (col + 4 <= res_K)                                                      \
                FETCH_FLOAT4(ldg_reg_a[(fg)][ld_id]) = FETCH_FLOAT4(A[row * K + col]); \
            else                                                                       \
                _Pragma("unroll") for (col; col < res_K; col++)                        \
                {                                                                      \
                    ldg_reg_a[(fg)][ld_id++] = A[row * K + col];                       \
                }                                                                      \
        }                                                                              \
    }                                                                                  \
    else                                                                               \
    {                                                                                  \
        int max_row = M - row_st;                                                      \
        _Pragma("unroll") for (int i = tid * 4; i < TILE_SIZE_A; i += NUM_THREADS * 4) \
        {                                                                              \
            int row = i / TILE_K;                                                      \
            if (row >= max_row)                                                        \
                break;                                                                 \
            int col = i % TILE_K;                                                      \
            int ld_id = (i / (NUM_THREADS * 4)) * 4;                                   \
            if (col + 4 <= res_K)                                                      \
                FETCH_FLOAT4(ldg_reg_a[(fg)][ld_id]) = FETCH_FLOAT4(A[row * K + col]); \
            else                                                                       \
                _Pragma("unroll") for (col; col < res_K; col++)                        \
                {                                                                      \
                    ldg_reg_a[(fg)][ld_id++] = A[row * K + col];                       \
                }                                                                      \
        }                                                                              \
    }                                                                                  \
    A += TILE_K;

#define LOAD_TILE_GMEM_TO_REG_B(fg)                                                    \
    if (col_st + TILE_N <= N)                                                          \
    {                                                                                  \
        _Pragma("unroll") for (int i = tid * 4; i < TILE_SIZE_B; i += NUM_THREADS * 4) \
        {                                                                              \
            int row = i / TILE_N;                                                      \
            if (row >= res_K)                                                          \
                break;                                                                 \
            int col = i % TILE_N;                                                      \
            int ld_id = (i / (NUM_THREADS * 4)) * 4;                                   \
            FETCH_FLOAT4(ldg_reg_b[(fg)][ld_id]) = FETCH_FLOAT4(B[row * N + col]);     \
        }                                                                              \
    }                                                                                  \
    else                                                                               \
    {                                                                                  \
        int max_col = N - col_st;                                                      \
        _Pragma("unroll") for (int i = tid * 4; i < TILE_SIZE_B; i += NUM_THREADS * 4) \
        {                                                                              \
            int row = i / TILE_N;                                                      \
            if (row >= res_K)                                                          \
                break;                                                                 \
            int col = i % TILE_N;                                                      \
            int ld_id = (i / (NUM_THREADS * 4)) * 4;                                   \
            if (col + 4 <= max_col)                                                    \
                FETCH_FLOAT4(ldg_reg_b[(fg)][ld_id]) = FETCH_FLOAT4(B[row * N + col]); \
            else                                                                       \
                _Pragma("unroll") for (col; col < max_col; col++)                      \
                {                                                                      \
                    ldg_reg_b[(fg)][ld_id++] = B[row * N + col];                       \
                }                                                                      \
        }                                                                              \
    }                                                                                  \
    B += TILE_K * N;

#define LOAD_TILE_REG_TO_SMEM_A(fg)                                                \
    _Pragma("unroll") for (int i = tid * 4; i < TILE_SIZE_A; i += NUM_THREADS * 4) \
    {                                                                              \
        int row = i / TILE_K;                                                      \
        int col = i % TILE_K;                                                      \
        int ld_id = (i / (NUM_THREADS * 4)) * 4;                                   \
        as[(fg)][col][row] = ldg_reg_a[(fg)][ld_id];                               \
        as[(fg)][col + 1][row] = ldg_reg_a[(fg)][ld_id + 1];                       \
        as[(fg)][col + 2][row] = ldg_reg_a[(fg)][ld_id + 2];                       \
        as[(fg)][col + 3][row] = ldg_reg_a[(fg)][ld_id + 3];                       \
    }
#define LOAD_TILE_REG_TO_SMEM_B(fg)                                                \
    _Pragma("unroll") for (int i = tid * 4; i < TILE_SIZE_B; i += NUM_THREADS * 4) \
    {                                                                              \
        int row = i / TILE_N;                                                      \
        int col = i % TILE_N;                                                      \
        int ld_id = (i / (NUM_THREADS * 4)) * 4;                                   \
        FETCH_FLOAT4(bs[(fg)][row][col]) = FETCH_FLOAT4(ldg_reg_b[(fg)][ld_id]);   \
    }

#define LOAD_FRAG_SMEM_TO_REG_A(ffg, k)                                            \
    _Pragma("unroll") for (int j = 0; j < FRAG_M; j += 4)                          \
    {                                                                              \
        FETCH_FLOAT4(ar[(ffg)][j]) = FETCH_FLOAT4(as[fg][(k)][thread_row_st + j]); \
    }
#define LOAD_FRAG_SMEM_TO_REG_B(ffg, k)                                            \
    _Pragma("unroll") for (int j = 0; j < FRAG_N; j += 4)                          \
    {                                                                              \
        FETCH_FLOAT4(br[(ffg)][j]) = FETCH_FLOAT4(bs[fg][(k)][thread_col_st + j]); \
    }

        template <
            const int TILE_M,
            const int TILE_N,
            const int TILE_K,
            const int FRAG_M,
            const int FRAG_N>
        __global__ void Sgemm(
            float *__restrict__ A,
            float *__restrict__ B,
            float *__restrict__ C,
            const int M, const int N, const int K)
        {
            // 当前线程负责的 frag 在 tile 中的起始行列值
            const int thread_col_st = threadIdx.x * FRAG_N;
            const int thread_row_st = threadIdx.y * FRAG_M;
            const int tid = threadIdx.y * blockDim.x + threadIdx.x;
            // 当前线程块负责的 tile 在全局的起始行列值
            const int col_st = blockIdx.x * TILE_N;
            const int row_st = blockIdx.y * TILE_M;
            A += row_st * K;
            B += col_st;
            // 当前线程块扫描的中间维度剩余长度
            int res_K = K;

            // tile 的大小
            constexpr int TILE_SIZE_A = TILE_M * TILE_K;
            constexpr int TILE_SIZE_B = TILE_K * TILE_N;
            constexpr int NUM_THREADS = (TILE_M / FRAG_M) * (TILE_N / FRAG_N);
            // 使用 smem 缓存 tile, 并预取下一个 tile
            int fg = 0;
            __shared__ float as[2][TILE_K][TILE_M]; // 转置，方便后续向量化访问
            __shared__ float bs[2][TILE_K][TILE_N];
            // tile 先从 gmem 加载到 reg, 再加载到 smem
            float ldg_reg_a[2][TILE_SIZE_A / NUM_THREADS] = {0};
            float ldg_reg_b[2][TILE_SIZE_B / NUM_THREADS] = {0};

            // 使用 reg 累加每个线程计算结果，最后再写回 gmem
            float accum[FRAG_M][FRAG_N] = {0};
            // 使用 reg 缓存 frag, 并预取下一个 frag
            int ffg = 0;
            float ar[2][FRAG_M] = {0};
            float br[2][FRAG_N] = {0};

            const int NUM_TILES = (K + TILE_K - 1) / TILE_K;
            int RES_FRAGS = min(TILE_K, res_K);
            // 加载第一个 tile 到 smem
            // 先加载到寄存器
            LOAD_TILE_GMEM_TO_REG_A(fg);
            LOAD_TILE_GMEM_TO_REG_B(fg);
            // 再加载到 smem
            LOAD_TILE_REG_TO_SMEM_A(fg);
            LOAD_TILE_REG_TO_SMEM_B(fg);

// 遍历所有 tile
#pragma unroll
            for (int i = 0; i < NUM_TILES; i++)
            {
                RES_FRAGS = min(TILE_K, res_K);
                __syncthreads();
                // 预取下一个 tile 到 reg
                if (i + 1 < NUM_TILES)
                {
                    res_K -= TILE_K;
                    LOAD_TILE_GMEM_TO_REG_A(fg ^ 1);
                    LOAD_TILE_GMEM_TO_REG_B(fg ^ 1);
                }

                // 先加载第一个 frag 到 reg
                LOAD_FRAG_SMEM_TO_REG_A(ffg, 0);
                LOAD_FRAG_SMEM_TO_REG_B(ffg, 0);
// 遍历所有 frag
#pragma unroll
                for (int j = 0; j < RES_FRAGS; j++)
                {
                    // 预取下一个 frag 到 reg
                    if (j + 1 < RES_FRAGS)
                    {
                        LOAD_FRAG_SMEM_TO_REG_A(ffg ^ 1, j + 1);
                        LOAD_FRAG_SMEM_TO_REG_B(ffg ^ 1, j + 1);
                    }
// 乘累加当前 frag
#pragma unroll
                    for (int y = 0; y < FRAG_M; y++)
                    {
#pragma unroll
                        for (int x = 0; x < FRAG_N; x++)
                        {
                            accum[y][x] += ar[ffg][y] * br[ffg][x];
                        }
                    }

                    ffg ^= 1;
                }

                // 加载下一个 tile 到 smem
                if (i + 1 < NUM_TILES)
                {
                    fg ^= 1;
                    LOAD_TILE_REG_TO_SMEM_A(fg);
                    LOAD_TILE_REG_TO_SMEM_B(fg);
                }
            }

            C += (row_st + thread_row_st) * N + col_st + thread_col_st;
            int res_row = min(FRAG_M, M - (row_st + thread_row_st));
            int res_col = min(FRAG_N, N - (col_st + thread_col_st));
// 将累加结果拷贝到 gmem
#pragma unroll
            for (int i = 0; i < res_row; i++)
            {
#pragma unroll
                for (int j = 0; j < res_col; j++)
                {
                    C[i * N + j] = accum[i][j];
                }
            }
        }

        void gemm_kernel(base::Tensor *input0, base::Tensor *input1, base::Tensor *output, void *stream)
        {
            size_t M = input0->shape(-2);
            size_t K = input0->shape(-1);
            CHECK_EQ(K, input1->shape(-2));
            size_t N = input1->shape(-1);
            CHECK_EQ(M, output->shape(-2));
            CHECK_EQ(N, output->shape(-1));

            CHECK(input0->num_mats() == 1);
            CHECK(input1->num_mats() == 1);
            CHECK(output->num_mats() == 1);

            if (!stream)
                LOG(WARNING) << "gemm_kernel: using default stream";
            if (N <= 1024)
            {
                dim3 grid((N + BLOCK_TILE_N_S - 1) / BLOCK_TILE_N_S, (M + BLOCK_TILE_M_S - 1) / BLOCK_TILE_M_S);
                dim3 block(BLOCK_TILE_N_S / THREAD_TILE_N, BLOCK_TILE_M_S / THREAD_TILE_M);
                Sgemm<BLOCK_TILE_M_S, BLOCK_TILE_N_S, BLOCK_TILE_K, THREAD_TILE_M, THREAD_TILE_N>
                    <<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(input0->data(),
                                                                            input1->data(),
                                                                            output->data(),
                                                                            M, N, K);
            }
            else
            {
                dim3 grid((N + BLOCK_TILE_N_L - 1) / BLOCK_TILE_N_L, (M + BLOCK_TILE_M_L - 1) / BLOCK_TILE_M_L);
                dim3 block(BLOCK_TILE_N_L / THREAD_TILE_N, BLOCK_TILE_M_L / THREAD_TILE_M);
                Sgemm<BLOCK_TILE_M_L, BLOCK_TILE_N_L, BLOCK_TILE_K, THREAD_TILE_M, THREAD_TILE_N>
                    <<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(input0->data(),
                                                                            input1->data(),
                                                                            output->data(),
                                                                            M, N, K);
            }
        }
    }
}
