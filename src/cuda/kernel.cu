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

    if (row >= L) return;

    float m_prev = -INFINITY;
    float d_prev = 0.0f;

    /* dynamic shared memory for O, Q, K, V
    extern __shared__ float s_memory[];
    
    // divide shared memory
    float* s_O = s_memory;
    float* s_Q = &s_O[B*D];
    //float* s_K = &s_Q[B*D];
    //float* s_V = &s_K[B*D];

    for (int j = tid; j < D; j += blockDim.x) {
        s_O[j] = 0;
        s_Q[j] = Q[i * D + j];
    }
    __syncthreads(); // syncwarps later??*/
      
    for (int col = 0; col < L; col++) {

        float x = 0.0f;
        for (int k = 0; k < D; k++) {
            x += Q[row * D + k] * K[col * D + k];
        }
        x = x * scale;

        //float m_local = warp_reduce_max(x);
        float m = max(m_prev, x);

        float num = d_prev * expf(m_prev - m);
        float exp_el = expf(x - m);
        //float exp_sum = warp_reduce_sum(exp_el);
        float d = num + exp_el;

        float alpha = num / d;
        float beta  = exp_el / d;

        for (int j = 0; j < D; j ++) {
            O[row * D + j] = O[row * D + j] * alpha + beta * V[col * D + j];
        }

        m_prev = m;
        d_prev = d;
    }

    /*for (int j = tid; j < D; j += blockDim.x) {
        O[i * D + j] = s_O[j];
    }*/
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
    size_t shared_mem_size = 0/*2 * d_k * sizeof(float)*/;
    
    flash_attn<<<blocks, threads, shared_mem_size>>>(
        Q, K, V, O, seq_len, d_k, scale
    );
}