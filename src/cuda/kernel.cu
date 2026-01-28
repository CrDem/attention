const unsigned mask = 0xffffffff;
__inline__ __device__
float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = max(val, __shfl_down_sync(mask, val, offset));
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

const int B = 32; // assert B == warpsize for warp reduce
__global__ void flash_attn(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    const int L,
    const int D,
    const float scale
) {
    const int numWarps = blockDim.y;
    const int tx = threadIdx.x; 
    const int ty = threadIdx.y;
    const int row = blockIdx.x * numWarps + ty;
    const bool row_valid = row < L;

    float m_prev = -INFINITY;
    float d_prev = 0.0f;

    extern __shared__ float s_mem[];

    float* s_O = s_mem;
    float* s_Q = &s_O[numWarps*D];
    float* s_K = &s_Q[numWarps*D];
    float* s_V = &s_K[B*D];
    float* s_A = &s_V[B*D];

    // load Q, zeroing O
    const int rowBias = ty * D;
    for (int j = tx; j < D; j += B) {
        s_O[rowBias + j] = 0.0f;
        s_Q[rowBias + j] = row_valid ? Q[row * D + j] : 0.0f;
    }
    __syncthreads();

    const int num_tiles = (L + B - 1) / B;
    for (int tile = 0; tile < num_tiles; tile++) {

        int num_cols_valid = min(B, L - tile * B);

        int col = tile * B + tx;
        bool col_valid = col < L;

        // load K, V
        for (int j = ty; j < D; j += numWarps) {
            s_K[tx * D + j] = col_valid ? K[col * D + j] : 0.0f;
            s_V[tx * D + j] = col_valid ? V[col * D + j] : 0.0f;
        }
        __syncthreads();

        // QK
        float x = -INFINITY;
        if (col_valid) {
            float acc = 0.0f;
            for (int k = 0; k < D; k++) {
                acc += s_Q[rowBias + k] * s_K[tx * D + k];
            }
            x = acc * scale;
        }

        // softmax
        float m_local = warp_reduce_max(x);
        float m = max(m_prev, m_local);
        m = __shfl_sync(mask, m, 0);

        float num = (d_prev > 0.0f) ? d_prev * expf(m_prev - m) : 0.0f;

        float exp_val = col_valid ? expf(x - m) : 0.0f;
        float exp_sum = warp_reduce_sum(exp_val);

        float d = num + exp_sum;
        d = __shfl_sync(mask, d, 0);

        s_A[ty * B + tx] = col_valid ? exp_val / d : 0.0f;
        __syncthreads();

        // O update
        float alpha = num / d;
        for (int j = tx; j < D; j += B) {
            float acc = 0.0f;
            for (int c = 0; c < num_cols_valid; c++) {
                acc += s_A[ty * B + c] * s_V[c * D + j];
            }
            s_O[rowBias + j] = s_O[rowBias + j] * alpha + acc;
        }

        m_prev = m;
        d_prev = d;

        __syncthreads(); // можно удалить если грузить V в smem между первым и вторым синком, а не вместе с К
        // в таком случае получим 2 синка вместо трех в цикле, но раздельную загрузку К и V
        // по перфу получается как будто одно и то же, так что оставил совместную загрузку для читаемости
    }

    // store O
    for (int j = tx; j < D; j += B) {
        if (row_valid) O[row * D + j] = s_O[rowBias + j];
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
    float scale,
    int num_warps) {
    
    dim3 block_size(B, num_warps);
    int blocks = batch_size * num_heads * ((seq_len + num_warps - 1) / num_warps);  // Total Q rows
    
    size_t shared_mem_size = (num_warps * d_k * 2 + B * d_k * 2 + B * num_warps) * sizeof(float);
    
    flash_attn<<<blocks, block_size, shared_mem_size>>>(
        Q, K, V, O, seq_len, d_k, scale
    );
}
