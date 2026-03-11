#include <cuda_fp16.h>
#include <mma.h>
#include <assert.h>
#include <math_constants.h>
#include <stdio.h>

using namespace nvcuda;

const unsigned fullMask = 0xffffffff;

const int Bc = 32;
const int Bc_pad = 40;

__global__ void flash_attn_bc32(
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
    const int Br = blockDim.y * 8;

    int warpColId = ty % 2;
    int warpRowId = ty / 2;
    const int warpColBase = warpColId * 16;
    const int warpRowBase = warpRowId * 16;

    extern __shared__ float s_mem[];

    float* s_O = s_mem;
    float* s_m_prev = &s_O[Br * D];
    float* s_d_prev = &s_m_prev[Br];
    float* s_row_max = &s_d_prev[Br];
    float* s_row_sum = &s_row_max[Br * 2];
    
    __half* s_A = (__half*)(((uintptr_t)&s_row_sum[Br * 2] + 15) & ~15); // ensure 16B alignment on half matrices
    __half* s_Q = (__half*)(((uintptr_t)&s_A[Br * Bc_pad] + 15) & ~15); 
    __half* s_K = (__half*)(((uintptr_t)&s_Q[Br * D_pad] + 15) & ~15);
    __half* s_V = (__half*)(((uintptr_t)&s_K[Bc * D_pad] + 15) & ~15);

    // wmma fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a1_frag;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a2_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b1_frag; // col major
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

    // aliases for code readability
    auto& q_frag = a1_frag;
    auto& k_frag = b1_frag;

    auto& v_frag = b1_frag;

    // init m_prev, d_prev
    if (tx < 8) {
        s_m_prev[ty * 8 + tx] = -CUDART_INF_F;;
        s_d_prev[ty * 8 + tx] = 0.0f;
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
        
        for (int d = 0; d < D; d += 32) {
            // 1st half
            wmma::load_matrix_sync(q_frag, &s_Q[warpRowBase * D_pad + d], D_pad);
            wmma::load_matrix_sync(k_frag, &s_K[warpColBase * D_pad + d], D_pad);
            wmma::mma_sync(acc_frag, q_frag, k_frag, acc_frag);

            // 2nd half
            wmma::load_matrix_sync(q_frag, &s_Q[warpRowBase * D_pad + d + 16], D_pad);
            wmma::load_matrix_sync(k_frag, &s_K[warpColBase * D_pad + d + 16], D_pad);
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

        // lanes 0,4,8,..,28 store the row-wise maximum values
        int row0 = warpRowBase + (tx / 4);
        int row1 = row0 + 8;
        if (tx % 4 == 0) {
            s_row_max[row0 * 2 + warpColId] = local_max0;
            s_row_max[row1 * 2 + warpColId] = local_max1;
        }
        __syncthreads();

        float m_row0 = fmaxf(s_row_max[row0 * 2 + 0], s_row_max[row0 * 2 + 1]);
        float m_row1 = fmaxf(s_row_max[row1 * 2 + 0], s_row_max[row1 * 2 + 1]);
        m_row0 = fmaxf(m_row0, s_m_prev[row0]);
        m_row1 = fmaxf(m_row1, s_m_prev[row1]);

        float exp_sum0 = 0.0f;
        float exp_sum1 = 0.0f;
        for (int i = 0; i < acc_frag.num_elements; i++) {
            bool colMask = i >= 4;
            int aColId = warpColBase + colMask * 8 + (tx % 4) * 2 + (i % 2);
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
        if (tx % 4 == 0) {
            s_row_sum[row0 * 2 + warpColId] = exp_sum0;
            s_row_sum[row1 * 2 + warpColId] = exp_sum1;
        }
        __syncthreads();

        float num0 = (s_d_prev[row0] > 0.0f) ? s_d_prev[row0] * expf(s_m_prev[row0] - m_row0) : 0.0f;
        float num1 = (s_d_prev[row1] > 0.0f) ? s_d_prev[row1] * expf(s_m_prev[row1] - m_row1) : 0.0f;

        float d0 = num0 + s_row_sum[row0 * 2 + 0] + s_row_sum[row0 * 2 + 1];
        float d1 = num1 + s_row_sum[row1 * 2 + 0] + s_row_sum[row1 * 2 + 1];

        if (tx % 4 == 0 && warpColId == 0) {
            s_d_prev[row0] = d0;
            s_m_prev[row0] = m_row0;
            s_d_prev[row1] = d1;
            s_m_prev[row1] = m_row1;
        }

        // 4. ------------ O scaling ------------
        for (int j = warpColId * 4 + tx % 4; j < D; j += 8) {
            s_O[row0 * D + j] = s_O[row0 * D + j] * num0 / d0;
            s_O[row1 * D + j] = s_O[row1 * D + j] * num1 / d1;
        }

        // 5. ------------ O += A x V ------------
        // store exp to s_A
        for (int i = 0; i < acc_frag.num_elements; i++)
        {
            bool colMask = i >= 4;
            int aColId = warpColBase + colMask * 8 + (tx % 4) * 2 + (i % 2);
            if (aColId >= aTileNumColsValid) continue;

            bool rowMask = (i >> 1) & 1; // {0, 1, 2, 3, 4, 5, 6, 7} -> {0, 0, 1, 1, 0, 0, 1, 1}

            float value = rowMask ? acc_frag.x[i] / d1 : acc_frag.x[i] / d0;

            int row = warpRowBase + rowMask * 8 + (tx / 4);
            s_A[row * Bc_pad + aColId] = __float2half(value);
        }
        __syncthreads();
        
        // load from s_A
        wmma::load_matrix_sync(a1_frag, &s_A[warpRowBase * Bc_pad], Bc_pad);
        wmma::load_matrix_sync(a2_frag, &s_A[warpRowBase * Bc_pad + 16], Bc_pad);

        for (int j = 0; j < D; j += Bc) {
            // load current O
            wmma::load_matrix_sync(acc_frag, &s_O[warpRowBase * D + j + warpColBase], D, wmma::mem_row_major);

            // 1st half
            wmma::load_matrix_sync(v_frag, &s_V[(j + warpColBase) * Bc_pad], Bc_pad);
            wmma::mma_sync(acc_frag, a1_frag, v_frag, acc_frag);

            // 2nd half
            wmma::load_matrix_sync(v_frag, &s_V[(j + warpColBase) * Bc_pad + 16], Bc_pad);
            wmma::mma_sync(acc_frag, a2_frag, v_frag, acc_frag);
            
            wmma::store_matrix_sync(&s_O[warpRowBase * D + j + warpColBase], acc_frag, D, wmma::mem_row_major);
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