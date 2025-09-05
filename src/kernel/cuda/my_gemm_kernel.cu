
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
            const int BLOCK_SIZE_M,         // height of block of C that each thread block calculate
            const int BLOCK_SIZE_K,         // width of block of A that each thread block load into shared memory
            const int BLOCK_SIZE_N,         // width of block of C that each thread block calculate
            const int THREAD_SIZE_Y,        // height of block of C that each thread calculate
            const int THREAD_SIZE_X,        // width of block of C that each thread calculate
            const bool ENABLE_DOUBLE_BUFFER // whether enable double buffering or not
            >
        __global__ void Sgemm(
            float *__restrict__ A,
            float *__restrict__ B,
            float *__restrict__ C,
            const int M,
            const int N,
            const int K)
        {
            // Block index
            int bx = blockIdx.x;
            int by = blockIdx.y;

            // Thread index
            int tx = threadIdx.x;
            int ty = threadIdx.y;

            // the threads number in Block of X,Y
            const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_X;
            const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;
            const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;

            // thread id in cur Block
            const int tid = ty * THREAD_X_PER_BLOCK + tx;

            // shared memory
            __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
            __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];
            // registers for C
            float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
            // registers for A and B
            float frag_a[2][THREAD_SIZE_Y];
            float frag_b[2][THREAD_SIZE_X];
            // registers load global memory
            const int ldg_num_a = BLOCK_SIZE_M * BLOCK_SIZE_K / (THREAD_NUM_PER_BLOCK * 4);
            const int ldg_num_b = BLOCK_SIZE_K * BLOCK_SIZE_N / (THREAD_NUM_PER_BLOCK * 4);
            float ldg_a_reg[4 * ldg_num_a];
            float ldg_b_reg[4 * ldg_num_b];

            // threads number in one row
            const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
            const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

            // row number and col number that needs to be loaded by this thread
            const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
            const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

            const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4;
            const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;

            // row stride that thread uses to load multiple rows of a tile
            const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
            const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

            A = &A[(BLOCK_SIZE_M * by) * K];
            B = &B[BLOCK_SIZE_N * bx];

// transfer first tile from global mem to shared mem
//  load A from global memory to shared memory
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE)
            {
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
                    A_TILE_ROW_START + i, // row
                    A_TILE_COL,           // col
                    K)]);
                As[0][A_TILE_COL][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index];
                As[0][A_TILE_COL + 1][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 1];
                As[0][A_TILE_COL + 2][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 2];
                As[0][A_TILE_COL + 3][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 3];
            }
// load B from global memory to shared memory
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE)
            {
                FETCH_FLOAT4(Bs[0][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
                    B_TILE_ROW_START + i, // row
                    B_TILE_COL,           // col
                    N)]);
            }
            __syncthreads();
// load A from shared memory to register
#pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4)
            {
                FETCH_FLOAT4(frag_a[0][thread_y]) = FETCH_FLOAT4(As[0][0][THREAD_SIZE_Y * ty + thread_y]);
            }
// load B from shared memory to register
#pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4)
            {
                FETCH_FLOAT4(frag_b[0][thread_x]) = FETCH_FLOAT4(Bs[0][0][THREAD_SIZE_X * tx + thread_x]);
            }

            int write_stage_idx = 1;
            int tile_idx = 0;
            do
            {
                tile_idx += BLOCK_SIZE_K;
                // load next tile from global mem
                if (tile_idx < K)
                {
#pragma unroll
                    for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE)
                    {
                        int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                        FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
                            A_TILE_ROW_START + i,  // row
                            A_TILE_COL + tile_idx, // col
                            K)]);
                    }
#pragma unroll
                    for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE)
                    {
                        int ldg_index = i / B_TILE_ROW_STRIDE * 4;
                        FETCH_FLOAT4(ldg_b_reg[ldg_index]) = FETCH_FLOAT4(B[OFFSET(
                            tile_idx + B_TILE_ROW_START + i, // row
                            B_TILE_COL,                      // col
                            N)]);
                    }
                }

                int load_stage_idx = write_stage_idx ^ 1;

#pragma unroll
                for (int j = 0; j < BLOCK_SIZE_K - 1; ++j)
                {
// load next tile from shared mem to register
// load A from shared memory to register
#pragma unroll
                    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4)
                    {
                        FETCH_FLOAT4(frag_a[(j + 1) % 2][thread_y]) = FETCH_FLOAT4(As[load_stage_idx][j + 1][THREAD_SIZE_Y * ty + thread_y]);
                    }
// load B from shared memory to register
#pragma unroll
                    for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4)
                    {
                        FETCH_FLOAT4(frag_b[(j + 1) % 2][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx][j + 1][THREAD_SIZE_X * tx + thread_x]);
                    }
// compute C THREAD_SIZE_X x THREAD_SIZE_Y
#pragma unroll
                    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y)
                    {
#pragma unroll
                        for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x)
                        {
                            accum[thread_y][thread_x] += frag_a[j % 2][thread_y] * frag_b[j % 2][thread_x];
                        }
                    }
                }

                if (tile_idx < K)
                {
#pragma unroll
                    for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE)
                    {
                        int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                        As[write_stage_idx][A_TILE_COL][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index];
                        As[write_stage_idx][A_TILE_COL + 1][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 1];
                        As[write_stage_idx][A_TILE_COL + 2][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 2];
                        As[write_stage_idx][A_TILE_COL + 3][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 3];
                    }
// load B from global memory to shared memory
#pragma unroll
                    for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE)
                    {
                        int ldg_index = i / B_TILE_ROW_STRIDE * 4;
                        FETCH_FLOAT4(Bs[write_stage_idx][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(ldg_b_reg[ldg_index]);
                    }
                    // use double buffer, only need one sync
                    __syncthreads();
                    // switch
                    write_stage_idx ^= 1;
                }

// load first tile from shared mem to register of next iter
// load A from shared memory to register
#pragma unroll
                for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4)
                {
                    FETCH_FLOAT4(frag_a[0][thread_y]) = FETCH_FLOAT4(As[load_stage_idx ^ 1][0][THREAD_SIZE_Y * ty + thread_y]);
                }
// load B from shared memory to register
#pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4)
                {
                    FETCH_FLOAT4(frag_b[0][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx ^ 1][0][THREAD_SIZE_X * tx + thread_x]);
                }
// compute last tile mma THREAD_SIZE_X x THREAD_SIZE_Y
#pragma unroll
                for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y)
                {
#pragma unroll
                    for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x)
                    {
                        accum[thread_y][thread_x] += frag_a[1][thread_y] * frag_b[1][thread_x];
                    }
                }
            } while (tile_idx < K);

// store back to C
#pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y)
            {
#pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4)
                {
                    FETCH_FLOAT4(C[OFFSET(
                        BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y,
                        BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x,
                        N)]) = FETCH_FLOAT4(accum[thread_y][thread_x]);
                }
            }
        }
        __global__ void sgemm(int M, int N, int K,
                              float *__restrict__ A,
                              float *__restrict__ B,
                              float *__restrict__ C)
        {
            const int tx = threadIdx.x;
            const int ty = threadIdx.y;
            const int tid = tx + ty * blockDim.x;
            // 线程块负责的行和列起点
            const int ROW = blockIdx.y * BLOCK_TILE_M_L;
            const int COL = blockIdx.x * BLOCK_TILE_N_L;
            A += ROW * K;
            B += COL;

            // 转移到smem的A、B TILE
            __shared__ float as[BLOCK_TILE_K][BLOCK_TILE_M_L];
            __shared__ float bs[BLOCK_TILE_K][BLOCK_TILE_N_L];
            // 转移到reg的A、B TILE
            float ar[THREAD_TILE_M];
            float br[THREAD_TILE_N];
            float accum[THREAD_TILE_M][THREAD_TILE_N] = {0};

            const int NUM_THREADS = (BLOCK_TILE_M_L / THREAD_TILE_M) * (BLOCK_TILE_N_L / THREAD_TILE_N);
            // 将一个tile转移到smem，每个线程需要加载的float4数量
            const int NUM_F4_LDG_A = (BLOCK_TILE_M_L * BLOCK_TILE_K / 4) / NUM_THREADS;
            const int NUM_F4_LDG_B = (BLOCK_TILE_N_L * BLOCK_TILE_K / 4) / NUM_THREADS;

            // 从gmem到smem的中转寄存器
            float buf_ldg_a[NUM_F4_LDG_A * 4];
            float buf_ldg_b[NUM_F4_LDG_B * 4];

            const int TILE_ROW_START_A = tid / (BLOCK_TILE_K / 4);
            const int TILE_ROW_START_B = tid / (BLOCK_TILE_N_L / 4);
            const int TILE_COL_A = (tid % (BLOCK_TILE_K / 4)) * 4;
            const int TILE_COL_B = (tid % (BLOCK_TILE_N_L / 4)) * 4;
            const int TILE_ROW_STRIDE_A = NUM_THREADS / (BLOCK_TILE_K / 4);
            const int TILE_ROW_STRIDE_B = NUM_THREADS / (BLOCK_TILE_N_L / 4);
#pragma unroll
            for (int i = 0; i < K; i += BLOCK_TILE_K)
            {
// gmem -> smem
#pragma unroll
                for (int j = 0; j < NUM_F4_LDG_A * 4; j += 4)
                {
                    int row = TILE_ROW_START_A + j * TILE_ROW_STRIDE_A;
                    FETCH_FLOAT4(buf_ldg_a[j]) = FETCH_FLOAT4(A[row * K + i + TILE_COL_A]);
                    as[TILE_COL_A][row] = buf_ldg_a[j];
                    as[TILE_COL_A + 1][row] = buf_ldg_a[j + 1];
                    as[TILE_COL_A + 2][row] = buf_ldg_a[j + 2];
                    as[TILE_COL_A + 3][row] = buf_ldg_a[j + 3];
                }
#pragma unroll
                for (int j = 0; j < NUM_F4_LDG_B * 4; j += 4)
                {
                    int row = TILE_ROW_START_B + j * TILE_ROW_STRIDE_B;
                    FETCH_FLOAT4(buf_ldg_b[j]) = FETCH_FLOAT4(B[(i + row) * N + TILE_COL_B]);
                    bs[row][TILE_COL_B] = buf_ldg_b[j];
                    bs[row][TILE_COL_B + 1] = buf_ldg_b[j + 1];
                    bs[row][TILE_COL_B + 2] = buf_ldg_b[j + 2];
                    bs[row][TILE_COL_B + 3] = buf_ldg_b[j + 3];
                }
                __syncthreads();

#pragma unroll
                for (int j = 0; j < BLOCK_TILE_K; j++)
                {
// smem -> reg
#pragma unroll
                    for (int k = 0; k < THREAD_TILE_M; k += 4)
                    {
                        int row = ty * THREAD_TILE_M + k;
                        FETCH_FLOAT4(ar[k]) = FETCH_FLOAT4(as[j][row]); // as 已经转置
                    }
#pragma unroll
                    for (int k = 0; k < THREAD_TILE_N; k += 4)
                    {
                        int col = tx * THREAD_TILE_N + k;
                        FETCH_FLOAT4(br[k]) = FETCH_FLOAT4(bs[j][col]);
                    }
// 计算
#pragma unroll
                    for (int y = 0; y < THREAD_TILE_M; y++)
                    {
                        for (int x = 0; x < THREAD_TILE_N; x++)
                        {
                            accum[y][x] += ar[y] * br[x];
                        }
                    }
                }
                __syncthreads();
            }

            // 将结果累加回全局内存
            C += (ROW + ty * THREAD_TILE_M) * N + COL + tx * THREAD_TILE_N;
#pragma unroll
            for (int i = 0; i < THREAD_TILE_M; i++)
            {
#pragma unroll
                for (int j = 0; j < THREAD_TILE_N; j += 4)
                {
                    FETCH_FLOAT4(C[i * N + j]) = FETCH_FLOAT4(accum[i][j]);
                }
            }
        }

        // #define TILE_M 128
        // #define TILE_N 128
        // #define TILE_K 8
        // #define FRAG_M 8
        // #define FRAG_N 8
        //         __global__ void sgemm_new(int M, int N, int K,
        //                                   float *__restrict__ A,
        //                                   float *__restrict__ B,
        //                                   float *__restrict__ C)
        //         {

        //         }

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
            CHECK(M % 8 == 0 && N % 8 == 0 && K % 8 == 0);

            if (!stream)
                LOG(WARNING) << "gemm_kernel: using default stream";
            if (N <= 1024)
            {
                dim3 grid(N / BLOCK_TILE_N_S, M / BLOCK_TILE_M_S);
                dim3 block(BLOCK_TILE_N_S / THREAD_TILE_N, BLOCK_TILE_M_S / THREAD_TILE_M);
                Sgemm<BLOCK_TILE_M_S, BLOCK_TILE_K, BLOCK_TILE_N_S, THREAD_TILE_M, THREAD_TILE_N, true>
                    <<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(input0->data(),
                                                                            input1->data(),
                                                                            output->data(),
                                                                            M, N, K);
            }
            else
            {
                dim3 grid(N / BLOCK_TILE_N_L, M / BLOCK_TILE_M_L);
                dim3 block(BLOCK_TILE_N_L / THREAD_TILE_N, BLOCK_TILE_M_L / THREAD_TILE_M);
                Sgemm<BLOCK_TILE_M_L, BLOCK_TILE_K, BLOCK_TILE_N_L, THREAD_TILE_M, THREAD_TILE_N, true>
                    <<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(input0->data(),
                                                                            input1->data(),
                                                                            output->data(),
                                                                            M, N, K);
            }
        }
    }
}
