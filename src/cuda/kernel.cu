#include <cuda_fp16.h>
#include <mma.h>
#include <assert.h>
#include <math_constants.h>

using namespace nvcuda;

const unsigned fullMask = 0xffffffff;

__inline__ __device__
float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(fullMask, val, offset));
    }
    return val;
}

__inline__ __device__
float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(fullMask, val, offset);
    }
    return val;
}

__global__ void flash_attn(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ O,
    const int L,
    const int D,
    const int Bc,
    const float scale
) {
    const int Bc_pad = Bc + 8;
    const int warpsPerRow = Bc / 16;

    const int D_pad = D + 8;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    const int numWarps = blockDim.y;
    const int Br = blockDim.y / warpsPerRow * 16;

    int warpColId = ty % warpsPerRow;
    int warpRowId = ty / warpsPerRow;
    const int warpColBase = warpColId * 16;
    const int warpRowBase = warpRowId * 16;

    extern __shared__ float s_mem[];

    float* s_O = s_mem;
    float* s_m_prev = &s_O[Br * D];
    float* s_d_prev = &s_m_prev[Br];
    float* s_A = &s_d_prev[Br]; 
    __half* s_Q = (__half*)(((uintptr_t)&s_A[Br * Bc_pad] + 15) & ~15); // ensure 16B alignment on half matrices

    // init m_prev, d_prev
    {
        if (ty % warpsPerRow == 0 && tx < 16) {
            s_m_prev[(ty / warpsPerRow) * 16 + tx] = -CUDART_INF_F;
            s_d_prev[(ty / warpsPerRow) * 16 + tx] = 0.0f;
        }
    }
    // load Q, zeroing O
    for (int localRow = ty; localRow < Br; localRow += numWarps) {
        int globalRow = blockIdx.x * Br + localRow;
        #pragma unroll 2
        for (int d = tx; d < D; d += 32) {
            s_O[localRow * D + d] = 0.0f;
            s_Q[localRow * D_pad + d] = globalRow < L ? Q[globalRow * D + d] : __float2half(0.0f);
        }
    }
    __syncthreads();
    
    const int num_tiles = (L + Bc - 1) / Bc;
    for (int tile = 0; tile < num_tiles; tile++) {
        int aTileCol = tile * Bc;
        int aTileNumColsValid = min(Bc, L - tile * Bc);

        {
            // wmma fragments
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> k_frag; // col major
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

            // 1. ------------ s_A <- Q x K ------------
            wmma::fill_fragment(acc_frag, 0.0f);
            
            #pragma unroll 4
            for (int d = 0; d < D; d += 16) { // D - inner dim ( Q[Br, D] x K[D, Bc] )
                wmma::load_matrix_sync(q_frag, &s_Q[warpRowBase * D_pad + d], D_pad);
                wmma::load_matrix_sync(k_frag, &K[(aTileCol + warpColBase) * D + d], D);
                wmma::mma_sync(acc_frag, q_frag, k_frag, acc_frag);
            }

            // 2. ----- store from wmma frag -----
            for (int i = 0; i < acc_frag.num_elements; i++) {
                bool colMask = i >= 4;
                int aColId = warpColBase + colMask * 8 + (tx % 4) * 2 + (i % 2);
                if (aColId >= aTileNumColsValid) continue;

                float v = acc_frag.x[i] * scale;

                bool rowMask = (i >> 1) & 1; // {0, 1, 2, 3, 4, 5, 6, 7} -> {0, 0, 1, 1, 0, 0, 1, 1}


                int row = warpRowBase + rowMask * 8 + (tx / 4);
                s_A[row * Bc_pad + aColId] = v;
            }
        }
        __syncthreads();

        // 4. softmax and O scaling
        for (int aRow = ty; aRow < Br; aRow += numWarps) {

            float m;
            {
                float m_thread = -CUDART_INF_F;
                for (int xId = 0; xId < Bc / 32; xId++) {
                    if (xId * 32 + tx >= aTileNumColsValid) break;
                    float v = s_A[aRow * Bc_pad + xId * 32 + tx];
                    m_thread = fmaxf(m_thread, v);
                }
                float m_local = warp_reduce_max(m_thread);
                m = fmaxf(s_m_prev[aRow], m_local);
            }
            m = __shfl_sync(fullMask, m, 0);

            float num = (s_d_prev[aRow] > 0.0f) ? s_d_prev[aRow] * expf(s_m_prev[aRow] - m) : 0.0f;

            float d;
            {
            float exp_thread = 0.0f;
            for (int xId = 0; xId < Bc / 32; xId++) {
                if (xId * 32 + tx >= aTileNumColsValid) break;
                float v = s_A[aRow * Bc_pad + xId * 32 + tx];
                exp_thread += expf(v - m);
            }
            float exp_sum = warp_reduce_sum(exp_thread);

            d = num + exp_sum;
            }
            d = __shfl_sync(fullMask, d, 0);
            if (tx == 0) {
                s_d_prev[aRow] = d;
                s_m_prev[aRow] = m;
            }

            // O scaling
            for (int j = tx; j < D; j += 32) {
                s_O[aRow * D + j] = s_O[aRow * D + j] * num / d;
            }

            // store exp/d to s_A
            #pragma unroll 1 
            for (int xId = 0; xId < Bc / 32; xId++) {
                if (xId * 32 + tx >= aTileNumColsValid) s_A[aRow * Bc_pad + xId * 32 + tx] = 0.0f;
                else {
                    float v = s_A[aRow * Bc_pad + xId * 32 + tx];
                    s_A[aRow * Bc_pad + xId * 32 + tx] = expf(v - m) / d;
                }
            }
        }
        __syncthreads();
        
        // 5. O += A x V
        {   
            // wmma fragments
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    
            for (int j = 0; j < D; j += Bc) { // if not enough warps (Bc < D)
                if (j + warpColBase >= D) break; // if too many warps (Bc > D or Bc % D != 0)
                // load current O (already scaled)
                wmma::load_matrix_sync(acc_frag, &s_O[warpRowBase * D + j + warpColBase], D, wmma::mem_row_major);

                #pragma unroll 4
                for (int bc_now = 0; bc_now < Bc; bc_now += 16) { // Bc - inner dim ( A[Br, Bc] x V[Bc, D] )
                    // ----- load to wmma frag -----
                    #pragma unroll 8
                    for (int i = 0; i < 8; i++) {

                        bool colMask = i >= 4;
                        int aColId = bc_now + colMask * 8 + (tx % 4) * 2 + (i % 2);
                        if (aColId >= aTileNumColsValid) {
                            a_frag.x[i] = __float2half(0.0f);
                            continue;
                        }

                        bool rowMask = (i >> 1) & 1; // {0, 1, 2, 3, 4, 5, 6, 7} -> {0, 0, 1, 1, 0, 0, 1, 1}
                        int row = warpRowBase + rowMask * 8 + (tx / 4);
                        a_frag.x[i] = __float2half(s_A[row * Bc_pad + aColId]);
                    }

                    // we can read out of V buffer on last tiles bc of zeroing corresponding elements in a_frag 
                    // (acc_frag_dot_product = acc_frag_dot_product + a_frag_el (0) * v_frag_el = acc_frag_dot_product)
                    wmma::load_matrix_sync(v_frag, &V[(aTileCol + bc_now) * D + (j + warpColBase)], D);
                    wmma::mma_sync(acc_frag, a_frag, v_frag, acc_frag);
                }
                
                wmma::store_matrix_sync(&s_O[warpRowBase * D + j + warpColBase], acc_frag, D, wmma::mem_row_major);
            }
        }
        __syncthreads();
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