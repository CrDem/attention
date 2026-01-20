/*__inline__ __device__
float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    return __shfl_sync(0xffffffff, val, 0);
}

__inline__ __device__
float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return __shfl_sync(0xffffffff, val, 0);
} */

const int B = 32;
const int TILE_SIZE = 1; //=b later
__global__ void flash_attn(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    const int L,
    const int D,
    const float scale
) {
    const int row = blockIdx.x * 32 + threadIdx.x; // Q row
    const int tid = threadIdx.x;

    if (row >= L) return;

    float m_prev = -INFINITY;
    float d_prev = 0.0f;

    // dynamic shared memory for O, Q, K, V
    extern __shared__ float s_memory[];
    
    // divide shared memory
    float* s_O = s_memory;
    float* s_Q = &s_O[32*D];
    float* s_X = &s_Q[32*D];
    float* s_VS = &s_X[32*32];
    //float* s_V = &s_K[B*D];

    for (int j = 0; j < D; j++) {
        s_O[tid * D + j] = 0.0f;
        s_Q[tid * D + j] = Q[row * D + j];
    }
    __syncthreads(); // syncwarps later??
      
    for (int tile = 0; tile < (L+31)/32; tile++) {
        for (int t = 0; t < 32; t++) {
            s_X[tid * 32 + t] = 0.0f;
        }
        for (int j = 0; j < D; j++) {
            s_VS[tid * D + j] = 0.0f;
        }
        for (int t = 0; t < 32 && ((tile * 32 + t) < L); t++) {
            for (int k = 0; k < D; k++) {
                s_X[tid * 32 + t] += s_Q[tid * D + k] * K[(tile * 32 + t) * D + k];
            }
        }
        for (int t = 0; t < 32 && ((tile * 32 + t) < L); t++) {
            s_X[tid * 32 + t] *= scale;
        }

        float m_local = -INFINITY;
        for (int t = 0; t < 32 && ((tile * 32 + t) < L); t++) {
            m_local = max(m_local, s_X[tid * 32 + t]);
        }
        float m = max(m_prev, m_local);

        float num = 0.0f;
        if (d_prev != 0) {
            num = d_prev * expf(m_prev - m);
        }
        float exp_sum = 0.0f;
        for (int t = 0; t < 32 && ((tile * 32 + t) < L); t++) {
            exp_sum += expf(s_X[tid * 32 + t] - m);
        }
        float d = num + exp_sum;

        float alpha = num / d;
        for (int t = 0; t < 32 && ((tile * 32 + t) < L); t++) {
            for (int j = 0; j < D; j++) {
                s_VS[tid * D + j] += (expf(s_X[tid * 32 + t] - m) / d) * V[(tile * 32 + t) * D + j];
            }
        }

        for (int j = 0; j < D; j++) {
            s_O[tid * D + j] = s_O[tid * D + j] * alpha + s_VS[tid * D + j];
        }

        m_prev = m;
        d_prev = d;
    }

    for (int j = 0; j < D; j++) {
        O[row * D + j] = s_O[tid * D + j];
    }
}

extern "C" void flash_attention_launcher(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int batch_size,
    int num_heads,
    int seq_len,
    int d_k,
    float scale) {
    
    // each block processes one Q row (i)
    int threads = B;  // 32 threads per block
    int blocks = batch_size * num_heads * ((seq_len + 31) / 32);  // Total Q rows
    
    // Shared memory calculation: O + Q = 2*d_k floats
    size_t shared_mem_size = ((3 * 32 * d_k) + 32*32) * sizeof(float);
    
    flash_attn<<<blocks, threads, shared_mem_size>>>(
        Q, K, V, O, seq_len, d_k, scale
    );
}
