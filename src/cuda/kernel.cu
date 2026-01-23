__inline__ __device__
float warp_reduce_max(float val, int width) {
    // width must be in [1, 32]
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        if (threadIdx.x + offset < width) {
            val = max(val, other);
        }
    }
    return val;
}

__inline__ __device__
float warp_reduce_sum(float val, int width) {
    // width must be in [1, 32]
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        if (threadIdx.x + offset < width)
            val += other;
    }
    return val;
}

const int B = 32; // assert B <= warpsize for warp reduce
const int numWarps = 32;
__global__ void flash_attn(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    const int L,
    const int D,
    const float scale
) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y; // Q row
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const bool row_valid = row < L;

    float m_prev = -INFINITY;
    float d_prev = 0.0f;

    // dynamic shared memory for O, Q, K, V
    extern __shared__ float s_memory[];
    
    // divide shared memory
    float* s_O = s_memory;
    float* s_Q = &s_O[numWarps*D];
    float* s_K = &s_Q[numWarps*D];
    //float* s_V = &s_K[B*D];
    float* s_VS = &s_K[B*D];

    const int rowBias = ty * D;
    for (int j = tx; j < D; j += B) {
        s_O[rowBias + j] = 0.0f;
        s_Q[rowBias + j] = row_valid ? Q[row * D + j] : 0.0f;
    }
    __syncthreads();
      
    for (int tile = 0; tile < (L + B - 1) / B; tile++) {
        int col = tile * B + tx;
        bool col_valid = col < L;
        
        float x = 0.0f;
        for (int j = tx; j < D; j += B) {
            s_VS[rowBias + j] = 0.0f;
        }

        for (int j = ty; j < D; j += numWarps) { // TODO: coalesced access if possible
            s_K[tx * D + j] = col_valid ? K[col * D + j] : 0.0f;
        }
        __syncthreads();
        
        if (col_valid) {
            for (int k = 0; k < D; k++) {
                x += s_Q[rowBias + k] * s_K[tx * D + k];
            }
            x *= scale;
        }

        float m_local = warp_reduce_max(x, B);
        float m = max(m_prev, m_local);
        m = __shfl_sync(0xffffffff, m, 0);

        float num = 0.0f;
        if (d_prev != 0) {
            num = d_prev * expf(m_prev - m);
        }
        float exp_val = 0.0f;
        if (col_valid) {
            exp_val = expf(x - m);
        }
        float exp_sum = warp_reduce_sum(exp_val, B);
        float d = num + exp_sum;
        d = __shfl_sync(0xffffffff, d, 0);

        if (col_valid) {
            float scalar = (expf(x - m) / d);
            for (int j = 0; j < D; j++) {
                atomicAdd(&s_VS[rowBias + j], scalar * V[(tile * B + tx) * D + j]);
            }
        }
        __syncthreads();
        
        float alpha = num / d;
        for (int j = tx; j < D; j += B) {
            s_O[rowBias + j] = s_O[rowBias + j] * alpha + s_VS[rowBias + j];
        }

        m_prev = m;
        d_prev = d;
    }

    if (row_valid) {
        for (int j = tx; j < D; j += B) {
            O[row * D + j] = s_O[rowBias + j];
        }
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
    
    dim3 block_size(B, numWarps);
    int blocks = batch_size * num_heads * ((seq_len + numWarps - 1) / numWarps);  // Total Q rows
    
    size_t shared_mem_size = (numWarps * d_k * 3 + B * d_k) * sizeof(float);
    
    flash_attn<<<blocks, block_size, shared_mem_size>>>(
        Q, K, V, O, seq_len, d_k, scale
    );
}
