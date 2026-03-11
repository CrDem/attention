#include <cuda_fp16.h>
#include <mma.h>
#include <assert.h>
#include <math_constants.h>

using namespace nvcuda;

const unsigned fullMask = 0xffffffff;

const int Bc = 16;
const int Bc_pad = 18;

__global__ void flash_attn_bc16(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ O,
    const int L,
    const int D,
    const float scale
) {
    const int D_pad = D + 8;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    const int numWarps = blockDim.y;
    const int Br = blockDim.y * 16 * (16 / Bc);

    const int warpRowBase = ty * 16;

    extern __shared__ float s_mem[];

    float* s_O = s_mem;
    float* s_m_prev = &s_O[Br * D];
    float* s_d_prev = &s_m_prev[Br];
    __half* s_Q = (__half*)(((uintptr_t)&s_d_prev[Br] + 15) & ~15); // ensure 16B alignment on half matrices
    __half* s_K = (__half*)(((uintptr_t)&s_Q[Br * D_pad] + 15) & ~15);
    __half* s_V = (__half*)(((uintptr_t)&s_K[Bc * D_pad] + 15) & ~15);

    // wmma fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag; // col major
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

    // aliases for code readability
    auto& q_frag = a_frag;
    auto& k_frag = b_frag;

    auto& v_frag = b_frag;

    // init m_prev, d_prev
    if (tx < 16) {
        s_m_prev[ty * 16 + tx] = -CUDART_INF_F;;
        s_d_prev[ty * 16 + tx] = 0.0f;
    }
    // load Q, zeroing O
    for (int localRow = ty; localRow < Br; localRow += numWarps) {
        int globalRow = blockIdx.x * Br + localRow;
        for (int d = tx; d < D; d += 32) {
            s_O[localRow * D + d] = 0.0f;
            s_Q[localRow * D_pad + d] = globalRow < L ? Q[globalRow * D + d] : __float2half(0.0f);
        }
    }

    const int num_tiles = (L + Bc - 1) / Bc;
    for (int tile = 0; tile < num_tiles; tile++) {
        int aTileCol = tile * Bc;
        int aTileNumColsValid = min(Bc, L - tile * Bc);

        // 1. ------------ load K, V ------------
        for (int localRow = ty; localRow < Bc; localRow += numWarps) {
            int globalRow = aTileCol + localRow;
            for (int d = tx; d < D; d += 32) {
                s_K[localRow * D_pad + d] = globalRow < L ? K[globalRow * D + d] : __float2half(0.0f);
                s_V[d * Bc_pad + localRow] = globalRow < L ? V[globalRow * D + d] : __float2half(0.0f); // transpose
            }
        }
        __syncthreads();

        // 2. ------------ s_A <- Q x K ------------
        wmma::fill_fragment(acc_frag, 0.0f);
        
        for (int d = 0; d < D; d += Bc) {
            wmma::load_matrix_sync(q_frag, &s_Q[warpRowBase * D_pad + d], D_pad);
            wmma::load_matrix_sync(k_frag, &s_K[d], D_pad);
            wmma::mma_sync(acc_frag, q_frag, k_frag, acc_frag);
        }

        // 3. ------------ softmax ------------
        float local_max0 = -CUDART_INF_F; // rows from 0 to 7
        float local_max1 = -CUDART_INF_F; // rows from 8 to 15

        for (int i = 0; i < acc_frag.num_elements; i++) {
            float v = acc_frag.x[i] * scale;

            bool rowMask = (i >> 1) & 1; // {0, 1, 2, 3, 4, 5, 6, 7} -> {0, 0, 1, 1, 0, 0, 1, 1}
            
            if (!rowMask)
                local_max0 = fmaxf(local_max0, v);
            else
                local_max1 = fmaxf(local_max1, v);
        }

        // reduce every 4 lanes
        for (int offset = 2; offset > 0; offset >>= 1) {
            local_max0 = fmaxf(local_max0, __shfl_down_sync(fullMask, local_max0, offset));
            local_max1 = fmaxf(local_max1, __shfl_down_sync(fullMask, local_max1, offset));
        }

        // each thread works with 2 rows
        int row0 = warpRowBase + (tx / 4);
        int row1 = row0 + 8;

        // lanes 0,4,8,..,28 store the row-wise maximum values
        // lane 0 -> lanes 0-3, lane 4 -> lanes 4-7, ect..
        int leader = tx & ~3; // 0,4,8,12,..
        local_max0 = __shfl_sync(0xffffffff, local_max0, leader);
        local_max1 = __shfl_sync(0xffffffff, local_max1, leader);

        // global max
        float m_row0 = fmaxf(local_max0, s_m_prev[row0]);
        float m_row1 = fmaxf(local_max1, s_m_prev[row1]);

        float exp_sum0 = 0.0f;
        float exp_sum1 = 0.0f;
        for (int i = 0; i < acc_frag.num_elements; i++) {
            bool colMask = i >= 4;
            int aColId = colMask * 8 + (tx % 4) * 2 + (i % 2);
            if (aColId >= aTileNumColsValid) continue;

            float v = acc_frag.x[i] * scale;

            bool rowMask = (i >> 1) & 1; // {0, 1, 2, 3, 4, 5, 6, 7} -> {0, 0, 1, 1, 0, 0, 1, 1}

            float m_curr = rowMask ? m_row1 : m_row0;
            float e = expf(v - m_curr);
            if (rowMask)
                exp_sum1 += e;
            else {
                exp_sum0 += e;
            }
            acc_frag.x[i] = e;
        }
        // reduce
        for (int offset = 2; offset > 0; offset >>= 1)
        {
            exp_sum0 += __shfl_down_sync(fullMask, exp_sum0, offset);
            exp_sum1 += __shfl_down_sync(fullMask, exp_sum1, offset);
        }

        // lanes 0,4,8,..,28 store the row-wise correct values
        float num0, num1, d0, d1;
        if (tx % 4 == 0) {
            num0 = (s_d_prev[row0] > 0.0f) ? s_d_prev[row0] * expf(s_m_prev[row0] - m_row0) : 0.0f;
            num1 = (s_d_prev[row1] > 0.0f) ? s_d_prev[row1] * expf(s_m_prev[row1] - m_row1) : 0.0f;
            d0 = num0 + exp_sum0;
            d1 = num1 + exp_sum1;
        }
        // lane 0 -> lanes 0-3, lane 4 -> lanes 4-7, ect..
        num0 = __shfl_sync(0xffffffff, num0, leader);
        num1 = __shfl_sync(0xffffffff, num1, leader);
        d0 = __shfl_sync(0xffffffff, d0, leader);
        d1 = __shfl_sync(0xffffffff, d1, leader);

        // store new m & d
        if (tx % 4 == 0) {
            s_d_prev[row0] = d0;
            s_m_prev[row0] = m_row0;
            s_d_prev[row1] = d1;
            s_m_prev[row1] = m_row1;
        }

        // 4. ------------ O scaling ------------
        for (int j = tx % 4; j < D; j += 4) {
            s_O[row0 * D + j] = s_O[row0 * D + j] * num0 / d0;
            s_O[row1 * D + j] = s_O[row1 * D + j] * num1 / d1;
        }

        // 5. ------------ O += A x V ------------
        // store exp to a_frag
        for (int i = 0; i < acc_frag.num_elements; i++)
        {
            bool colMask = i >= 4;
            int aColId = colMask * 8 + (tx % 4) * 2 + (i % 2);
            if (aColId >= aTileNumColsValid) continue;

            bool rowMask = (i >> 1) & 1; // {0, 1, 2, 3, 4, 5, 6, 7} -> {0, 0, 1, 1, 0, 0, 1, 1}

            float value = rowMask ? acc_frag.x[i] / d1 : acc_frag.x[i] / d0;

            a_frag.x[i] = __float2half(value);
        }

        for (int j = 0; j < D; j += Bc) {
            // load current O
            wmma::load_matrix_sync(acc_frag, &s_O[warpRowBase * D + j], D, wmma::mem_row_major);

            wmma::load_matrix_sync(v_frag, &s_V[j * Bc_pad], Bc_pad);
            wmma::mma_sync(acc_frag, a_frag, v_frag, acc_frag);
            
            wmma::store_matrix_sync(&s_O[warpRowBase * D + j], acc_frag, D, wmma::mem_row_major);
        }

        __syncthreads(); // можно удалить если грузить V в smem между первым и вторым синком, а не вместе с К
        // в таком случае получим 2 синка вместо трех в цикле, но раздельную загрузку К и V
        // по перфу получается как будто одно и то же, так что оставил совместную загрузку для читаемости
    }

    // s_O -> O
    for (int localRow = ty; localRow < Br; localRow += numWarps) {
        int globalRow = blockIdx.x * Br + localRow;
        if (globalRow < L) {
            for (int d = tx; d < D; d += 32) {
                O[globalRow * D + d] = __float2half(s_O[localRow * D + d]);
            }
        }
    }
}