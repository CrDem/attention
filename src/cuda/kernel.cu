#include <cuda_fp16.h>
#include <mma.h>
#include <assert.h>
#include <math_constants.h>

using namespace nvcuda;

const unsigned mask = 0xffffffff;

__inline__ __device__
float warp_reduce_max(float val) {  
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(mask, val, offset));
    }
    return val;
}

__inline__ __device__
float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

const int Bc = 32; // assert Bc == warpsize for warp reduce
const int Bc_pad = 40;
__global__ void flash_attn(
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
    
    const int localTileRow = ty / 2 * 16;
    const int localTileCol = ty % 2 * 16;

    extern __shared__ float s_mem[];

    float* s_O = s_mem;
    float* s_m_prev = &s_O[Br * D];
    float* s_d_prev = &s_m_prev[Br];
    float* s_A = &s_d_prev[Br];
    
    __half* s_A_half = (__half*)(((uintptr_t)&s_A[Br * Bc_pad] + 15) & ~15); // ensure 16B alignment on half matrices
    __half* s_Q = (__half*)(((uintptr_t)&s_A_half[Br * Bc_pad] + 15) & ~15); 
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
        bool aColValid = aTileCol + tx < L;

        // load K, V
        for (int localRow = ty; localRow < Bc; localRow += numWarps) {
            int globalRow = aTileCol + localRow;
            for (int d = tx; d < D; d += 32) {
                s_K[localRow * D_pad + d] = globalRow < L ? K[globalRow * D + d] : __float2half(0.0f);
                s_V[d * Bc_pad + localRow] = globalRow < L ? V[globalRow * D + d] : __float2half(0.0f); // transpose
            }
        }
        __syncthreads();

        // s_A <- Q x K
        wmma::fill_fragment(acc_frag, 0.0f);
        
        for (int d = 0; d < D; d += 32) {
            // 1st half
            wmma::load_matrix_sync(q_frag, &s_Q[localTileRow * D_pad + d], D_pad);
            wmma::load_matrix_sync(k_frag, &s_K[localTileCol * D_pad + d], D_pad);
            wmma::mma_sync(acc_frag, q_frag, k_frag, acc_frag);

            // 2nd half
            wmma::load_matrix_sync(q_frag, &s_Q[localTileRow * D_pad + d + 16], D_pad);
            wmma::load_matrix_sync(k_frag, &s_K[localTileCol * D_pad + d + 16], D_pad);
            wmma::mma_sync(acc_frag, q_frag, k_frag, acc_frag);
        }

        wmma::store_matrix_sync(&s_A[localTileRow * Bc_pad + localTileCol], acc_frag, Bc_pad, wmma::mem_row_major);
        __syncthreads();

        for (int aRow = ty; aRow < Br; aRow += numWarps) {
            float x = aColValid ? s_A[aRow * Bc_pad + tx] * scale : -CUDART_INF_F;

            // softmax
            float m_local = warp_reduce_max(x);
            float m = fmaxf(s_m_prev[aRow], m_local);
            m = __shfl_sync(mask, m, 0);

            float num = (s_d_prev[aRow] > 0.0f) ? s_d_prev[aRow] * expf(s_m_prev[aRow] - m) : 0.0f;

            float exp_val = aColValid ? expf(x - m) : 0.0f;
            float exp_sum = warp_reduce_sum(exp_val);

            float d = num + exp_sum;
            d = __shfl_sync(mask, d, 0);
            if (tx == 0) {
                s_d_prev[aRow] = d;
                s_m_prev[aRow] = m;
            }

            // O scaling
            float alpha;
            if (tx == 0) {
                alpha = num / d;
            }
            alpha = __shfl_sync(mask, alpha, 0);

            for (int j = tx; j < D; j += Bc) {
                s_O[aRow * D + j] = s_O[aRow * D + j] * alpha;
            }

            s_A_half[aRow * Bc_pad + tx] = aColValid ? __float2half(exp_val / d) : __float2half(0.0f);
        }
        __syncthreads();
        
        // O += A x V
        wmma::load_matrix_sync(a1_frag, &s_A_half[localTileRow * Bc_pad], Bc_pad);
        wmma::load_matrix_sync(a2_frag, &s_A_half[localTileRow * Bc_pad + 16], Bc_pad);

        for (int j = 0; j < D; j += Bc) {
            // load current O
            wmma::load_matrix_sync(acc_frag, &s_O[localTileRow * D + j + localTileCol], D, wmma::mem_row_major);

            // 1st half
            wmma::load_matrix_sync(v_frag, &s_V[(j + localTileCol) * Bc_pad], Bc_pad);
            wmma::mma_sync(acc_frag, a1_frag, v_frag, acc_frag);

            // 2nd half
            wmma::load_matrix_sync(v_frag, &s_V[(j + localTileCol) * Bc_pad + 16], Bc_pad);
            wmma::mma_sync(acc_frag, a2_frag, v_frag, acc_frag);
            
            wmma::store_matrix_sync(&s_O[localTileRow * D + j + localTileCol], acc_frag, D, wmma::mem_row_major);
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

extern "C" void flash_attention_launcher(
    const __half* Q,
    const __half* K,
    const __half* V,
    __half* O,
    int batch_size,
    int num_heads,
    int seq_len,
    int d_k,
    float scale,
    int Br
) {
    dim3 block_size(32, Br/8);
    int blocks = batch_size * num_heads * ((seq_len + Br - 1) / Br); // Total Q rows

    size_t shared_mem_size =
        (Br * d_k * sizeof(float)) +   // s_O 4*64*64 = 16kb
        (Bc_pad * Br * sizeof(float)) +    // s_A 40*64*4 = 10kb (+2)
        (Bc_pad * Br * sizeof(__half)) +   // s_A_half (padded) 40*64*2 = 5kb (+1)
        (Br * (d_k + 8) * sizeof(__half)) +  // s_Q (padded) 64*72*2 = 9kb (+1)
        (Bc * (d_k + 8) * sizeof(__half)) +  // s_K (padded) .. = 4.5kb (+0.5)
        (Bc_pad * d_k * sizeof(__half)) +    // s_V (padded) 40*64*2 = 5kb (+1)
        (Br * sizeof(float)) * 2;      // s_m_prev, s_d_prev 64*4*2 = 0.5kb
        // = 50kb

    cudaFuncSetAttribute(
        flash_attn,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem_size
    );
        
    flash_attn<<<blocks, block_size, shared_mem_size>>>(
        Q, K, V, O, seq_len, d_k, scale
    );
}
