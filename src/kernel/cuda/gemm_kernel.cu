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
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

        template <
            const int TILE_M,
            const int TILE_N,
            const int TILE_K,
            const int FRAG_M,
            const int FRAG_N>
        __global__ void SgemmDiv1(
            float *__restrict__ A,
            float *__restrict__ B,
            float *__restrict__ C,
            const int M, const int N, const int K)
        {
            const int tx = threadIdx.x;
            const int ty = threadIdx.y;
            const int tid = ty * blockDim.x + tx;
            const int bx = blockIdx.x;
            const int by = blockIdx.y;
            // 当前线程块负责的 tile 在全局的起始行列
            const int ROW_ST = by * TILE_M;
            const int COL_ST = bx * TILE_N;
            // 当前线程负责的 frag 在 tile 中的起始行列
            const int TH_ROW_ST = ty * FRAG_M;
            const int TH_COL_ST = tx * FRAG_N;
            // 移动到负责的 tile 位置
            A += ROW_ST * K;
            B += COL_ST;
            C += (ROW_ST + TH_ROW_ST) * N + COL_ST + TH_COL_ST;

            constexpr int TILE_SIZE_A = TILE_M * TILE_K;
            constexpr int TILE_SIZE_B = TILE_K * TILE_N;
            constexpr int NUM_THREADS = (TILE_M / FRAG_M) * (TILE_N / FRAG_N);

            // 每个线程负责加载 tile 的一部分所需的参数，每次加载单个元素
            constexpr int NUM_TH_PER_ROW_LD_A = TILE_K;
            constexpr int NUM_TH_PER_ROW_LD_B = TILE_N;
            const int LD_TILE_ROW_ST_A = tid / NUM_TH_PER_ROW_LD_A;
            const int LD_TILE_ROW_ST_B = tid / NUM_TH_PER_ROW_LD_B;
            const int LD_TILE_ROW_STRIDE_A = NUM_THREADS / NUM_TH_PER_ROW_LD_A;
            const int LD_TILE_ROW_STRIDE_B = NUM_THREADS / NUM_TH_PER_ROW_LD_B;
            const int LD_COL_TILE_A = tid % NUM_TH_PER_ROW_LD_A;
            const int LD_COL_TILE_B = tid % NUM_TH_PER_ROW_LD_B;

            // 缓存 tile 到 smem, 还要预取下一个 tile
            int fg = 0;
            __shared__ float as[2][TILE_K][TILE_M]; // 转置存储，方便后续向量化读取一列
            __shared__ float bs[2][TILE_K][TILE_N];
            // 转移 tile 的中转寄存器
            float ldg_reg_a[TILE_SIZE_A / NUM_THREADS] = {0};
            float ldg_reg_b[TILE_SIZE_B / NUM_THREADS] = {0};

            // 每个线程加载各自负责的 frag 到寄存器计算, 还要预取下一个 frag
            int ffg = 0;
            float ar[2][FRAG_M] = {0};
            float br[2][FRAG_N] = {0};
            // 每个线程负责的 frag 计算结果累加在寄存器中，最后再写回全局内存
            float accum[FRAG_M][FRAG_N] = {0};

            // 当前线程块实际负责的 tile 大小（实际矩阵大小不整除TILE_M，TILE_N）
            const int T_TILE_M = min(TILE_M, M - ROW_ST);
            const int T_TILE_N = min(TILE_N, N - COL_ST);
            // 遍历中间维度剩余长度
            int res_K = K;

// 加载第一个 tile 到 smem
// 先加载到 reg
#pragma unroll
            for (int i = 0; i < TILE_M; i += LD_TILE_ROW_STRIDE_A)
            {
                int ld_id = i / LD_TILE_ROW_STRIDE_A;
                if (LD_TILE_ROW_ST_A + i < T_TILE_M && LD_COL_TILE_A < res_K)
                {
                    ldg_reg_a[ld_id] = A[OFFSET(LD_TILE_ROW_ST_A + i, LD_COL_TILE_A, K)];
                }
                else
                {
                    ldg_reg_a[ld_id] = 0.0f;
                }
                as[fg][LD_COL_TILE_A][LD_TILE_ROW_ST_A + i] = ldg_reg_a[ld_id];
            }
            A += TILE_K;
#pragma unroll
            for (int i = 0; i < TILE_K; i += LD_TILE_ROW_STRIDE_B)
            {
                int ld_id = i / LD_TILE_ROW_STRIDE_B;
                if (LD_TILE_ROW_ST_B + i < res_K && LD_COL_TILE_B < T_TILE_N)
                {
                    ldg_reg_b[ld_id] = B[OFFSET(LD_TILE_ROW_ST_B + i, LD_COL_TILE_B, N)];
                }
                else
                {
                    ldg_reg_b[ld_id] = 0.0f;
                }
                bs[fg][LD_TILE_ROW_ST_B + i][LD_COL_TILE_B] = ldg_reg_b[ld_id];
            }
            B += TILE_K * N;
            res_K -= TILE_K;

            // 遍历所有 tile
            for (int i = 0; i < K; i += TILE_K)
            {
                __syncthreads();
// 加载第一个 frag 到寄存器
#pragma unroll
                for (int j = 0; j < FRAG_M; j++)
                    ar[ffg][j] = as[fg][0][TH_ROW_ST + j];
#pragma unroll
                for (int j = 0; j < FRAG_N; j++)
                    br[ffg][j] = bs[fg][0][TH_COL_ST + j];

                // 预取下一个 tile 到寄存器
                if (i + TILE_K < K)
                {
#pragma unroll
                    for (int ii = 0; ii < TILE_M; ii += LD_TILE_ROW_STRIDE_A)
                    {
                        int ld_id = ii / LD_TILE_ROW_STRIDE_A;
                        if (LD_TILE_ROW_ST_A + ii < T_TILE_M && LD_COL_TILE_A < res_K)
                        {
                            ldg_reg_a[ld_id] = A[OFFSET(LD_TILE_ROW_ST_A + ii, LD_COL_TILE_A, K)];
                        }
                        else
                        {
                            ldg_reg_a[ld_id] = 0.0f;
                        }
                    }
                    A += TILE_K;
#pragma unroll
                    for (int ii = 0; ii < TILE_K; ii += LD_TILE_ROW_STRIDE_B)
                    {
                        int ld_id = ii / LD_TILE_ROW_STRIDE_B;
                        if (LD_TILE_ROW_ST_B + ii < res_K && LD_COL_TILE_B < T_TILE_N)
                        {
                            ldg_reg_b[ld_id] = B[OFFSET(LD_TILE_ROW_ST_B + ii, LD_COL_TILE_B, N)];
                        }
                        else
                        {
                            ldg_reg_b[ld_id] = 0.0f;
                        }
                    }
                    B += TILE_K * N;
                    res_K -= TILE_K;
                }

// 遍历所有 frag
#pragma unroll
                for (int j = 0; j < TILE_K; j++)
                {
                    // 预取下一个 frag
                    if (j + 1 < TILE_K)
                    {
#pragma unroll
                        for (int k = 0; k < FRAG_M; k++)
                            ar[ffg ^ 1][k] = as[fg][j + 1][TH_ROW_ST + k];
#pragma unroll
                        for (int k = 0; k < FRAG_N; k++)
                            br[ffg ^ 1][k] = bs[fg][j + 1][TH_COL_ST + k];
                    }

// 乘累加结果到寄存器
#pragma unroll
                    for (int y = 0; y < FRAG_M; y++)
                    {
#pragma unroll
                        for (int x = 0; x < FRAG_N; x++)
                        {
                            accum[y][x] += ar[ffg][y] * br[ffg][x];
                        }
                    }

                    // 反转 frag 缓冲区标志位
                    ffg ^= 1;
                }

                // 加载下一个 tile 到 smem
                if (i + TILE_K < K)
                {
                    // 反转 tile 缓冲区标志位
                    fg ^= 1;
#pragma unroll
                    for (int ii = 0; ii < TILE_M; ii += LD_TILE_ROW_STRIDE_A)
                    {
                        int ld_id = ii / LD_TILE_ROW_STRIDE_A;
                        as[fg][LD_COL_TILE_A][LD_TILE_ROW_ST_A + ii] = ldg_reg_a[ld_id];
                    }
#pragma unroll
                    for (int ii = 0; ii < TILE_K; ii += LD_TILE_ROW_STRIDE_B)
                    {
                        int ld_id = ii / LD_TILE_ROW_STRIDE_B;
                        bs[fg][LD_TILE_ROW_ST_B + ii][LD_COL_TILE_B] = ldg_reg_b[ld_id];
                    }
                }
            }

            // 将累加的结果写回全局内存
#pragma unroll
            for (int i = 0; i < FRAG_M; i++)
            {
#pragma unroll
                for (int j = 0; j < FRAG_N; j++)
                {
                    if (TH_ROW_ST + i < T_TILE_M && TH_COL_ST + j < T_TILE_N)
                        C[i * N + j] = accum[i][j];
                }
            }
        }

        template <
            const int TILE_M,
            const int TILE_N,
            const int TILE_K,
            const int FRAG_M,
            const int FRAG_N>
        __global__ void SgemmDiv4(
            float *__restrict__ A,
            float *__restrict__ B,
            float *__restrict__ C,
            const int M, const int N, const int K)
        {
            const int tx = threadIdx.x;
            const int ty = threadIdx.y;
            const int tid = ty * blockDim.x + tx;
            const int bx = blockIdx.x;
            const int by = blockIdx.y;
            // 当前线程块负责的 tile 在全局的起始行列
            const int ROW_ST = by * TILE_M;
            const int COL_ST = bx * TILE_N;
            // 当前线程负责的 frag 在 tile 中的起始行列
            const int TH_ROW_ST = ty * FRAG_M;
            const int TH_COL_ST = tx * FRAG_N;
            // 移动到负责的 tile 位置
            A += ROW_ST * K;
            B += COL_ST;
            C += (ROW_ST + TH_ROW_ST) * N + COL_ST + TH_COL_ST;

            constexpr int TILE_SIZE_A = TILE_M * TILE_K;
            constexpr int TILE_SIZE_B = TILE_K * TILE_N;
            constexpr int NUM_THREADS = (TILE_M / FRAG_M) * (TILE_N / FRAG_N);

            // 每个线程负责加载 tile 的一部分所需的参数，每次使用 float4 加载
            constexpr int NUM_TH_PER_ROW_LD_A = TILE_K / 4;
            constexpr int NUM_TH_PER_ROW_LD_B = TILE_N / 4;
            const int LD_TILE_ROW_ST_A = tid / NUM_TH_PER_ROW_LD_A;
            const int LD_TILE_ROW_ST_B = tid / NUM_TH_PER_ROW_LD_B;
            const int LD_TILE_ROW_STRIDE_A = NUM_THREADS / NUM_TH_PER_ROW_LD_A;
            const int LD_TILE_ROW_STRIDE_B = NUM_THREADS / NUM_TH_PER_ROW_LD_B;
            const int LD_COL_TILE_A = (tid % NUM_TH_PER_ROW_LD_A) * 4;
            const int LD_COL_TILE_B = (tid % NUM_TH_PER_ROW_LD_B) * 4;

            // 缓存 tile 到 smem, 还要预取下一个 tile
            int fg = 0;
            __shared__ float as[2][TILE_K][TILE_M]; // 转置存储，方便后续向量化读取一列
            __shared__ float bs[2][TILE_K][TILE_N];
            // 转移 tile 的中转寄存器
            float ldg_reg_a[TILE_SIZE_A / NUM_THREADS] = {0};
            float ldg_reg_b[TILE_SIZE_B / NUM_THREADS] = {0};

            // 每个线程加载各自负责的 frag 到寄存器计算, 还要预取下一个 frag
            int ffg = 0;
            float ar[2][FRAG_M] = {0};
            float br[2][FRAG_N] = {0};
            // 每个线程负责的 frag 计算结果累加在寄存器中，最后再写回全局内存
            float accum[FRAG_M][FRAG_N] = {0};

            // 当前线程块实际负责的 tile 大小（实际矩阵大小不整除TILE_M，TILE_N）
            const int T_TILE_M = min(TILE_M, M - ROW_ST);
            const int T_TILE_N = min(TILE_N, N - COL_ST);
            // 遍历中间维度剩余长度
            int res_K = K;

// 加载第一个 tile 到 smem
// 先加载到 reg
#pragma unroll
            for (int i = 0; i < TILE_M; i += LD_TILE_ROW_STRIDE_A)
            {
                int ld_id = i / LD_TILE_ROW_STRIDE_A * 4;
                if (LD_TILE_ROW_ST_A + i < T_TILE_M)
                {
                    if (LD_COL_TILE_A + 3 < res_K)
                        FETCH_FLOAT4(ldg_reg_a[ld_id]) = FETCH_FLOAT4(A[OFFSET(LD_TILE_ROW_ST_A + i, LD_COL_TILE_A, K)]);
                    else
                    {
                        ldg_reg_a[ld_id] = LD_COL_TILE_A < res_K ? A[OFFSET(LD_TILE_ROW_ST_A + i, LD_COL_TILE_A, K)] : 0.0f;
                        ldg_reg_a[ld_id + 1] = LD_COL_TILE_A + 1 < res_K ? A[OFFSET(LD_TILE_ROW_ST_A + i, LD_COL_TILE_A + 1, K)] : 0.0f;
                        ldg_reg_a[ld_id + 2] = LD_COL_TILE_A + 2 < res_K ? A[OFFSET(LD_TILE_ROW_ST_A + i, LD_COL_TILE_A + 2, K)] : 0.0f;
                        ldg_reg_a[ld_id + 3] = 0.0f;
                    }
                }
                else
                    FETCH_FLOAT4(ldg_reg_a[ld_id]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                as[fg][LD_COL_TILE_A][LD_TILE_ROW_ST_A + i] = ldg_reg_a[ld_id];
                as[fg][LD_COL_TILE_A + 1][LD_TILE_ROW_ST_A + i] = ldg_reg_a[ld_id + 1];
                as[fg][LD_COL_TILE_A + 2][LD_TILE_ROW_ST_A + i] = ldg_reg_a[ld_id + 2];
                as[fg][LD_COL_TILE_A + 3][LD_TILE_ROW_ST_A + i] = ldg_reg_a[ld_id + 3];
            }
            A += TILE_K;
#pragma unroll
            for (int i = 0; i < TILE_K; i += LD_TILE_ROW_STRIDE_B)
            {
                if (LD_TILE_ROW_ST_B + i < res_K)
                {
                    if (LD_COL_TILE_B + 3 < T_TILE_N)
                        FETCH_FLOAT4(bs[fg][LD_TILE_ROW_ST_B + i][LD_COL_TILE_B]) = FETCH_FLOAT4(B[OFFSET(LD_TILE_ROW_ST_B + i, LD_COL_TILE_B, N)]);
                    else
                    {
                        bs[fg][LD_TILE_ROW_ST_B + i][LD_COL_TILE_B] = LD_COL_TILE_B < T_TILE_N ? B[OFFSET(LD_TILE_ROW_ST_B + i, LD_COL_TILE_B, N)] : 0.0f;
                        bs[fg][LD_TILE_ROW_ST_B + i][LD_COL_TILE_B + 1] = LD_COL_TILE_B + 1 < T_TILE_N ? B[OFFSET(LD_TILE_ROW_ST_B + i, LD_COL_TILE_B + 1, N)] : 0.0f;
                        bs[fg][LD_TILE_ROW_ST_B + i][LD_COL_TILE_B + 2] = LD_COL_TILE_B + 2 < T_TILE_N ? B[OFFSET(LD_TILE_ROW_ST_B + i, LD_COL_TILE_B + 2, N)] : 0.0f;
                        bs[fg][LD_TILE_ROW_ST_B + i][LD_COL_TILE_B + 3] = 0.0f;
                    }
                }
                else
                    FETCH_FLOAT4(bs[fg][LD_TILE_ROW_ST_B + i][LD_COL_TILE_B]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            B += TILE_K * N;
            res_K -= TILE_K;

            // 遍历所有 tile
            for (int i = 0; i < K; i += TILE_K)
            {
                __syncthreads();
// 加载第一个 frag 到寄存器
#pragma unroll
                for (int j = 0; j < FRAG_M; j += 4)
                    FETCH_FLOAT4(ar[ffg][j]) = FETCH_FLOAT4(as[fg][0][TH_ROW_ST + j]);
#pragma unroll
                for (int j = 0; j < FRAG_N; j += 4)
                    FETCH_FLOAT4(br[ffg][j]) = FETCH_FLOAT4(bs[fg][0][TH_COL_ST + j]);

                // 预取下一个 tile 到寄存器
                if (i + TILE_K < K)
                {
#pragma unroll
                    for (int i = 0; i < TILE_M; i += LD_TILE_ROW_STRIDE_A)
                    {
                        int ld_id = i / LD_TILE_ROW_STRIDE_A * 4;
                        if (LD_TILE_ROW_ST_A + i < T_TILE_M)
                        {
                            if (LD_COL_TILE_A + 3 < res_K)
                                FETCH_FLOAT4(ldg_reg_a[ld_id]) = FETCH_FLOAT4(A[OFFSET(LD_TILE_ROW_ST_A + i, LD_COL_TILE_A, K)]);
                            else
                            {
                                ldg_reg_a[ld_id] = LD_COL_TILE_A < res_K ? A[OFFSET(LD_TILE_ROW_ST_A + i, LD_COL_TILE_A, K)] : 0.0f;
                                ldg_reg_a[ld_id + 1] = LD_COL_TILE_A + 1 < res_K ? A[OFFSET(LD_TILE_ROW_ST_A + i, LD_COL_TILE_A + 1, K)] : 0.0f;
                                ldg_reg_a[ld_id + 2] = LD_COL_TILE_A + 2 < res_K ? A[OFFSET(LD_TILE_ROW_ST_A + i, LD_COL_TILE_A + 2, K)] : 0.0f;
                                ldg_reg_a[ld_id + 3] = 0.0f;
                            }
                        }
                        else
                            FETCH_FLOAT4(ldg_reg_a[ld_id]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    }
                    A += TILE_K;
#pragma unroll
                    for (int i = 0; i < TILE_K; i += LD_TILE_ROW_STRIDE_B)
                    {
                        int ld_id = i / LD_TILE_ROW_STRIDE_B * 4;
                        if (LD_TILE_ROW_ST_B + i < res_K)
                        {
                            if (LD_COL_TILE_B + 3 < T_TILE_N)
                                FETCH_FLOAT4(ldg_reg_b[ld_id]) = FETCH_FLOAT4(B[OFFSET(LD_TILE_ROW_ST_B + i, LD_COL_TILE_B, N)]);
                            else
                            {
                                ldg_reg_b[ld_id] = LD_COL_TILE_B < T_TILE_N ? B[OFFSET(LD_TILE_ROW_ST_B + i, LD_COL_TILE_B, N)] : 0.0f;
                                ldg_reg_b[ld_id + 1] = LD_COL_TILE_B + 1 < T_TILE_N ? B[OFFSET(LD_TILE_ROW_ST_B + i, LD_COL_TILE_B + 1, N)] : 0.0f;
                                ldg_reg_b[ld_id + 2] = LD_COL_TILE_B + 2 < T_TILE_N ? B[OFFSET(LD_TILE_ROW_ST_B + i, LD_COL_TILE_B + 2, N)] : 0.0f;
                                ldg_reg_b[ld_id + 3] = 0.0f;
                            }
                        }
                        else
                            FETCH_FLOAT4(ldg_reg_b[ld_id]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    }
                    B += TILE_K * N;
                    res_K -= TILE_K;
                }

// 遍历所有 frag
#pragma unroll
                for (int j = 0; j < TILE_K; j++)
                {
                    // 预取下一个 frag
                    if (j + 1 < TILE_K)
                    {
#pragma unroll
                        for (int k = 0; k < FRAG_M; k += 4)
                            FETCH_FLOAT4(ar[ffg ^ 1][k]) = FETCH_FLOAT4(as[fg][j + 1][TH_ROW_ST + k]);
#pragma unroll
                        for (int k = 0; k < FRAG_N; k += 4)
                            FETCH_FLOAT4(br[ffg ^ 1][k]) = FETCH_FLOAT4(bs[fg][j + 1][TH_COL_ST + k]);
                    }

// 乘累加结果到寄存器
#pragma unroll
                    for (int y = 0; y < FRAG_M; y++)
                    {
#pragma unroll
                        for (int x = 0; x < FRAG_N; x++)
                        {
                            accum[y][x] += ar[ffg][y] * br[ffg][x];
                        }
                    }

                    // 反转 frag 缓冲区标志位
                    ffg ^= 1;
                }

                // 加载下一个 tile 到 smem
                if (i + TILE_K < K)
                {
                    // 反转 tile 缓冲区标志位
                    fg ^= 1;
#pragma unroll
                    for (int ii = 0; ii < TILE_M; ii += LD_TILE_ROW_STRIDE_A)
                    {
                        int ld_id = (ii / LD_TILE_ROW_STRIDE_A) * 4;
                        as[fg][LD_COL_TILE_A][LD_TILE_ROW_ST_A + ii] = ldg_reg_a[ld_id];
                        as[fg][LD_COL_TILE_A + 1][LD_TILE_ROW_ST_A + ii] = ldg_reg_a[ld_id + 1];
                        as[fg][LD_COL_TILE_A + 2][LD_TILE_ROW_ST_A + ii] = ldg_reg_a[ld_id + 2];
                        as[fg][LD_COL_TILE_A + 3][LD_TILE_ROW_ST_A + ii] = ldg_reg_a[ld_id + 3];
                    }
#pragma unroll
                    for (int ii = 0; ii < TILE_K; ii += LD_TILE_ROW_STRIDE_B)
                    {
                        int ld_id = (ii / LD_TILE_ROW_STRIDE_B) * 4;
                        FETCH_FLOAT4(bs[fg][LD_TILE_ROW_ST_B + ii][LD_COL_TILE_B]) = FETCH_FLOAT4(ldg_reg_b[ld_id]);
                    }
                }
            }

            // 将累加的结果写回全局内存
            if (TH_ROW_ST + FRAG_M - 1 < T_TILE_M && TH_COL_ST + FRAG_N - 1 < T_TILE_N)
            {
#pragma unroll
                for (int i = 0; i < FRAG_M; i++)
                {
#pragma unroll
                    for (int j = 0; j < FRAG_N; j += 4)
                    {
                        FETCH_FLOAT4(C[i * N + j]) = FETCH_FLOAT4(accum[i][j]);
                    }
                }
            }
            else
            {
#pragma unroll
                for (int i = 0; i < FRAG_M; i++)
                {
#pragma unroll
                    for (int j = 0; j < FRAG_N; j++)
                    {
                        if (TH_ROW_ST + i < T_TILE_M && TH_COL_ST + j < T_TILE_N)
                            C[i * N + j] = accum[i][j];
                    }
                }
            }
        }

        template <
            const int TILE_M,
            const int TILE_N,
            const int TILE_K,
            const int FRAG_M,
            const int FRAG_N>
        __global__ void SgemmDiv64(
            float *__restrict__ A,
            float *__restrict__ B,
            float *__restrict__ C,
            const int M, const int N, const int K)
        {
            const int tx = threadIdx.x;
            const int ty = threadIdx.y;
            const int tid = ty * blockDim.x + tx;
            const int bx = blockIdx.x;
            const int by = blockIdx.y;
            // 当前线程块负责的 tile 在全局的起始行列
            const int ROW_ST = by * TILE_M;
            const int COL_ST = bx * TILE_N;
            // 当前线程负责的 frag 在 tile 中的起始行列
            const int TH_ROW_ST = ty * FRAG_M;
            const int TH_COL_ST = tx * FRAG_N;
            // 移动到负责的 tile 位置
            A += ROW_ST * K;
            B += COL_ST;
            C += (ROW_ST + TH_ROW_ST) * N + COL_ST + TH_COL_ST;

            constexpr int TILE_SIZE_A = TILE_M * TILE_K;
            constexpr int TILE_SIZE_B = TILE_K * TILE_N;
            constexpr int NUM_THREADS = (TILE_M / FRAG_M) * (TILE_N / FRAG_N);

            // 每个线程负责加载 tile 的一部分所需的参数，每次使用 float4 加载
            constexpr int NUM_TH_PER_ROW_LD_A = TILE_K / 4;
            constexpr int NUM_TH_PER_ROW_LD_B = TILE_N / 4;
            const int LD_TILE_ROW_ST_A = tid / NUM_TH_PER_ROW_LD_A;
            const int LD_TILE_ROW_ST_B = tid / NUM_TH_PER_ROW_LD_B;
            const int LD_TILE_ROW_STRIDE_A = NUM_THREADS / NUM_TH_PER_ROW_LD_A;
            const int LD_TILE_ROW_STRIDE_B = NUM_THREADS / NUM_TH_PER_ROW_LD_B;
            const int LD_COL_TILE_A = (tid % NUM_TH_PER_ROW_LD_A) * 4;
            const int LD_COL_TILE_B = (tid % NUM_TH_PER_ROW_LD_B) * 4;

            // 缓存 tile 到 smem, 还要预取下一个 tile
            int fg = 0;
            __shared__ float as[2][TILE_K][TILE_M]; // 转置存储，方便后续向量化读取一列
            __shared__ float bs[2][TILE_K][TILE_N];
            // 转移 tile 的中转寄存器
            float ldg_reg_a[TILE_SIZE_A / NUM_THREADS] = {0};
            float ldg_reg_b[TILE_SIZE_B / NUM_THREADS] = {0};

            // 每个线程加载各自负责的 frag 到寄存器计算, 还要预取下一个 frag
            int ffg = 0;
            float ar[2][FRAG_M] = {0};
            float br[2][FRAG_N] = {0};
            // 每个线程负责的 frag 计算结果累加在寄存器中，最后再写回全局内存
            float accum[FRAG_M][FRAG_N] = {0};

// 加载第一个 tile 到 smem
// 先加载到 reg
#pragma unroll
            for (int i = 0; i < TILE_M; i += LD_TILE_ROW_STRIDE_A)
            {
                int ld_id = i / LD_TILE_ROW_STRIDE_A * 4;
                FETCH_FLOAT4(ldg_reg_a[ld_id]) = FETCH_FLOAT4(A[OFFSET(LD_TILE_ROW_ST_A + i, LD_COL_TILE_A, K)]);
                as[fg][LD_COL_TILE_A][LD_TILE_ROW_ST_A + i] = ldg_reg_a[ld_id];
                as[fg][LD_COL_TILE_A + 1][LD_TILE_ROW_ST_A + i] = ldg_reg_a[ld_id + 1];
                as[fg][LD_COL_TILE_A + 2][LD_TILE_ROW_ST_A + i] = ldg_reg_a[ld_id + 2];
                as[fg][LD_COL_TILE_A + 3][LD_TILE_ROW_ST_A + i] = ldg_reg_a[ld_id + 3];
            }
            A += TILE_K;
#pragma unroll
            for (int i = 0; i < TILE_K; i += LD_TILE_ROW_STRIDE_B)
            {
                FETCH_FLOAT4(bs[fg][LD_TILE_ROW_ST_B + i][LD_COL_TILE_B]) = FETCH_FLOAT4(B[OFFSET(LD_TILE_ROW_ST_B + i, LD_COL_TILE_B, N)]);
            }
            B += TILE_K * N;

            // 遍历所有 tile
            for (int i = 0; i < K; i += TILE_K)
            {
                __syncthreads();
// 加载第一个 frag 到寄存器
#pragma unroll
                for (int j = 0; j < FRAG_M; j += 4)
                    FETCH_FLOAT4(ar[ffg][j]) = FETCH_FLOAT4(as[fg][0][TH_ROW_ST + j]);
#pragma unroll
                for (int j = 0; j < FRAG_N; j += 4)
                    FETCH_FLOAT4(br[ffg][j]) = FETCH_FLOAT4(bs[fg][0][TH_COL_ST + j]);

                // 预取下一个 tile 到寄存器
                if (i + TILE_K < K)
                {
#pragma unroll
                    for (int i = 0; i < TILE_M; i += LD_TILE_ROW_STRIDE_A)
                    {
                        int ld_id = i / LD_TILE_ROW_STRIDE_A * 4;
                        FETCH_FLOAT4(ldg_reg_a[ld_id]) = FETCH_FLOAT4(A[OFFSET(LD_TILE_ROW_ST_A + i, LD_COL_TILE_A, K)]);
                    }
                    A += TILE_K;
#pragma unroll
                    for (int i = 0; i < TILE_K; i += LD_TILE_ROW_STRIDE_B)
                    {
                        int ld_id = i / LD_TILE_ROW_STRIDE_B * 4;
                        FETCH_FLOAT4(ldg_reg_b[ld_id]) = FETCH_FLOAT4(B[OFFSET(LD_TILE_ROW_ST_B + i, LD_COL_TILE_B, N)]);
                    }
                    B += TILE_K * N;
                }

// 遍历所有 frag
#pragma unroll
                for (int j = 0; j < TILE_K; j++)
                {
                    // 预取下一个 frag
                    if (j + 1 < TILE_K)
                    {
#pragma unroll
                        for (int k = 0; k < FRAG_M; k += 4)
                            FETCH_FLOAT4(ar[ffg ^ 1][k]) = FETCH_FLOAT4(as[fg][j + 1][TH_ROW_ST + k]);
#pragma unroll
                        for (int k = 0; k < FRAG_N; k += 4)
                            FETCH_FLOAT4(br[ffg ^ 1][k]) = FETCH_FLOAT4(bs[fg][j + 1][TH_COL_ST + k]);
                    }

// 乘累加结果到寄存器
#pragma unroll
                    for (int y = 0; y < FRAG_M; y++)
                    {
#pragma unroll
                        for (int x = 0; x < FRAG_N; x++)
                        {
                            accum[y][x] += ar[ffg][y] * br[ffg][x];
                        }
                    }

                    // 反转 frag 缓冲区标志位
                    ffg ^= 1;
                }

                // 加载下一个 tile 到 smem
                if (i + TILE_K < K)
                {
                    // 反转 tile 缓冲区标志位
                    fg ^= 1;
#pragma unroll
                    for (int ii = 0; ii < TILE_M; ii += LD_TILE_ROW_STRIDE_A)
                    {
                        int ld_id = (ii / LD_TILE_ROW_STRIDE_A) * 4;
                        as[fg][LD_COL_TILE_A][LD_TILE_ROW_ST_A + ii] = ldg_reg_a[ld_id];
                        as[fg][LD_COL_TILE_A + 1][LD_TILE_ROW_ST_A + ii] = ldg_reg_a[ld_id + 1];
                        as[fg][LD_COL_TILE_A + 2][LD_TILE_ROW_ST_A + ii] = ldg_reg_a[ld_id + 2];
                        as[fg][LD_COL_TILE_A + 3][LD_TILE_ROW_ST_A + ii] = ldg_reg_a[ld_id + 3];
                    }
#pragma unroll
                    for (int ii = 0; ii < TILE_K; ii += LD_TILE_ROW_STRIDE_B)
                    {
                        int ld_id = (ii / LD_TILE_ROW_STRIDE_B) * 4;
                        FETCH_FLOAT4(bs[fg][LD_TILE_ROW_ST_B + ii][LD_COL_TILE_B]) = FETCH_FLOAT4(ldg_reg_b[ld_id]);
                    }
                }
            }

// 将累加的结果写回全局内存
#pragma unroll
            for (int i = 0; i < FRAG_M; i++)
            {
#pragma unroll
                for (int j = 0; j < FRAG_N; j += 4)
                {
                    FETCH_FLOAT4(C[i * N + j]) = FETCH_FLOAT4(accum[i][j]);
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
                dim3 block((BLOCK_TILE_N_S + THREAD_TILE_N - 1) / THREAD_TILE_N, (BLOCK_TILE_M_S + THREAD_TILE_M - 1) / THREAD_TILE_M);
                if (M % BLOCK_TILE_M_S == 0 && N % BLOCK_TILE_N_S == 0 && K % BLOCK_TILE_K == 0)
                    SgemmDiv64<BLOCK_TILE_M_S, BLOCK_TILE_N_S, BLOCK_TILE_K, THREAD_TILE_M, THREAD_TILE_N>
                        <<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(input0->data(),
                                                                                input1->data(),
                                                                                output->data(),
                                                                                M, N, K);
                else if (M % 4 == 0 && N % 4 == 0 && K % 4 == 0)
                    SgemmDiv4<BLOCK_TILE_M_S, BLOCK_TILE_N_S, BLOCK_TILE_K, THREAD_TILE_M, THREAD_TILE_N>
                        <<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(input0->data(),
                                                                                input1->data(),
                                                                                output->data(),
                                                                                M, N, K);
                else
                    SgemmDiv1<BLOCK_TILE_M_S, BLOCK_TILE_N_S, BLOCK_TILE_K, THREAD_TILE_M, THREAD_TILE_N>
                        <<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(input0->data(),
                                                                                input1->data(),
                                                                                output->data(),
                                                                                M, N, K);
            }
            else
            {
                dim3 grid((N + BLOCK_TILE_N_L - 1) / BLOCK_TILE_N_L, (M + BLOCK_TILE_M_L - 1) / BLOCK_TILE_M_L);
                dim3 block((BLOCK_TILE_N_L + THREAD_TILE_N - 1) / THREAD_TILE_N, (BLOCK_TILE_M_L + THREAD_TILE_M - 1) / THREAD_TILE_M);
                if (M % BLOCK_TILE_M_L == 0 && N % BLOCK_TILE_N_L == 0 && K % BLOCK_TILE_K == 0)
                    SgemmDiv64<BLOCK_TILE_M_L, BLOCK_TILE_N_L, BLOCK_TILE_K, THREAD_TILE_M, THREAD_TILE_N>
                        <<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(input0->data(),
                                                                                input1->data(),
                                                                                output->data(),
                                                                                M, N, K);
                else if (M % 4 == 0 && N % 4 == 0 && K % 4 == 0)
                    SgemmDiv4<BLOCK_TILE_M_L, BLOCK_TILE_N_L, BLOCK_TILE_K, THREAD_TILE_M, THREAD_TILE_N>
                        <<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(input0->data(),
                                                                                input1->data(),
                                                                                output->data(),
                                                                                M, N, K);
                else
                    SgemmDiv1<BLOCK_TILE_M_L, BLOCK_TILE_N_L, BLOCK_TILE_K, THREAD_TILE_M, THREAD_TILE_N>
                        <<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(input0->data(),
                                                                                input1->data(),
                                                                                output->data(),
                                                                                M, N, K);
            }
        }
    }
}
