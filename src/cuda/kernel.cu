#include <cuda_fp16.h>
#include <mma.h>
#include <assert.h>

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

const int Bc = 32; // assert B == warpsize for warp reduce
__global__ void flash_attn(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ O,
    const int L,
    const int D,
    const float scale
) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int Br = 32;
    const int localTileRow = ty / 2 * 16;
    const int localTileCol = ty % 2 * 16;

    extern __shared__ float s_mem[];

    float* s_O = s_mem;
    float* s_V_acc = &s_O[Br * D];
    float* s_m_prev = &s_V_acc[Br * D];
    float* s_d_prev = &s_m_prev[Br];
    float* s_num = &s_d_prev[Br];
    float* s_A = &s_num[Br];
    __half* s_A_half = (__half*)&s_A[Bc * Br];
    __half* s_Q = &s_A_half[Bc * Br];
    __half* s_K = &s_Q[Br * D];
    __half* s_V = &s_K[Bc * D];

    // wmma fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> k_frag; // col major
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

    // reuse q and k later
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a1_frag;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a2_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag; // row major
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> v_new_acc_frag;

    if (tx < 8) {
        s_m_prev[ty * 8 + tx] = -INFINITY;
        s_d_prev[ty * 8 + tx] = 0.0f;
    }
    // load Q, zeroing O
    for (int localRow = ty; localRow < Br; localRow += 4) {
        int globalRow = blockIdx.x * Br + localRow;
        for (int d = tx; d < D; d += 32) {
            s_O[localRow * D + d] = 0.0f;
            s_Q[localRow * D + d] = globalRow < L ? Q[globalRow * D + d] : __float2half(0.0f);
        }
    }

    const int num_tiles = (L + Bc - 1) / Bc;
    for (int tile = 0; tile < num_tiles; tile++) {
        int aTileCol = tile * Bc;
        int aTileNumColsValid = min(Bc, L - tile * Bc);
        int aCol = tx; // pos in warp
        bool aColValid = aTileCol + aCol < L;

        // load K, V
        for (int localRow = ty; localRow < Bc; localRow += 4) {
            int globalRow = aTileCol + localRow;
            for (int d = tx; d < D; d += 32) {
                s_K[localRow * D + d] = globalRow < L ? K[globalRow * D + d] : __float2half(0.0f);
                s_V[localRow * D + d] = globalRow < L ? V[globalRow * D + d] : __float2half(0.0f);
            }
        }
        __syncthreads();

        // s_Q[Br, D] * (s_K[Br, D])
        wmma::fill_fragment(acc_frag, 0.0f);
        
        for (int d = 0; d < D; d += 32) {
            // 1st half
            wmma::load_matrix_sync(q_frag, &s_Q[localTileRow * D + d], D);
            wmma::load_matrix_sync(k_frag, &s_K[localTileCol * D + d], D);
            wmma::mma_sync(acc_frag, q_frag, k_frag, acc_frag);

            // 2nd half
            wmma::load_matrix_sync(q_frag, &s_Q[localTileRow * D + d + 16], D);
            wmma::load_matrix_sync(k_frag, &s_K[localTileCol * D + d + 16], D);
            wmma::mma_sync(acc_frag, q_frag, k_frag, acc_frag);
        }

        wmma::store_matrix_sync(&s_A[localTileRow * Bc + localTileCol], acc_frag, 32, wmma::mem_row_major);
        __syncthreads();

        for (int aRow = ty; aRow < Br; aRow += 4) {
            float x = aColValid ? s_A[aRow * Bc + tx] * scale : -INFINITY;
            // softmax
            float m_local = warp_reduce_max(x);
            float m = fmaxf(s_m_prev[aRow], m_local);
            m = __shfl_sync(mask, m, 0);

            float num = (s_d_prev[aRow] > 0.0f) ? s_d_prev[aRow] * expf(s_m_prev[aRow] - m) : 0.0f;
            if (tx == 0) {
                s_num[aRow] = num;
            }

            float exp_val = aColValid ? expf(x - m) : 0.0f;
            float exp_sum = warp_reduce_sum(exp_val);

            float d = num + exp_sum;
            d = __shfl_sync(mask, d, 0);
            if (tx == 0) {
                s_d_prev[aRow] = d;
                s_m_prev[aRow] = m;
            }

            s_A_half[aRow * Bc + tx] = aColValid ? __float2half(exp_val / d) : __float2half(0.0f);
        }
        __syncthreads();

        // A x V
        wmma::load_matrix_sync(a1_frag, &s_A_half[localTileRow * Bc], Bc);
        wmma::load_matrix_sync(a2_frag, &s_A_half[localTileRow * Bc + 16], Bc);

        for (int j = 0; j < D; j += Bc) {
            wmma::fill_fragment(v_new_acc_frag, 0.0f);

            // 1st half
            wmma::load_matrix_sync(v_frag, &s_V[j + localTileCol], D);
            wmma::mma_sync(v_new_acc_frag, a1_frag, v_frag, v_new_acc_frag);

            // 2nd half
            wmma::load_matrix_sync(v_frag, &s_V[16 * D + j + localTileCol], D);
            wmma::mma_sync(v_new_acc_frag, a2_frag, v_frag, v_new_acc_frag);
            
            wmma::store_matrix_sync(&s_V_acc[localTileRow * D + j + localTileCol], v_new_acc_frag, D, wmma::mem_row_major);
        }
        __syncthreads();
        
        // O update
        for (int aRow = ty; aRow < Br; aRow += 4) { // 1 warp calculating 1 s_O row
            int rowBias = aRow * D;
            
            float alpha;
            if (aCol == 0) {
                alpha = s_num[aRow] / s_d_prev[aRow];
            }
            alpha = __shfl_sync(mask, alpha, 0);

            for (int j = aCol; j < D; j += Bc) {
                s_O[rowBias + j] = s_O[rowBias + j] * alpha + s_V_acc[rowBias + j];
            }
        }

        __syncthreads(); // можно удалить если грузить V в smem между первым и вторым синком, а не вместе с К
        // в таком случае получим 2 синка вместо трех в цикле, но раздельную загрузку К и V
        // по перфу получается как будто одно и то же, так что оставил совместную загрузку для читаемости
    }
    // load Q, zeroing O
    for (int localRow = ty; localRow < Br; localRow += 4) {
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
    int num_warps
) {
    dim3 block_size(32, 4);
    int blocks = batch_size * num_heads * ((seq_len + num_warps - 1) / num_warps); // Total Q rows

    size_t shared_mem_size =
        (num_warps * d_k * sizeof(float)) +   // s_O
        (num_warps * d_k * sizeof(float)) +   // s_V_acc
        (Bc * num_warps * sizeof(float)) +    // s_A
        (Bc * num_warps * sizeof(__half)) +   // s_A_half
        (num_warps * d_k * sizeof(__half)) +  // s_Q
        (Bc * d_k * sizeof(__half)) * 2 +     // s_K, s_V
        (32 * sizeof(float)) * 3;             // s_m_prev, s_d_prev, s_num 
        
    flash_attn<<<blocks, block_size, shared_mem_size>>>(
        Q, K, V, O, seq_len, d_k, scale
    );
}
