#include <cuda_runtime.h>

__global__ void qk_mul_kernel(
    const float* Q,
    const float* K,
    float* O,
    int batch_size,
    int num_heads,
    int seq_len,
    int d_k,
    float scale) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_heads * seq_len * seq_len;
    
    if (idx >= total_elements) return;
    
    // Разворачиваем индекс
    int n = idx / (num_heads * seq_len * seq_len);
    int h = (idx % (num_heads * seq_len * seq_len)) / (seq_len * seq_len);
    int i = (idx % (seq_len * seq_len)) / seq_len;
    int j = idx % seq_len;
    
    // Смещения
    int q_offset = n * num_heads * seq_len * d_k + h * seq_len * d_k + i * d_k;
    int k_offset = n * num_heads * seq_len * d_k + h * seq_len * d_k + j * d_k;
    
    // Скалярное произведение
    float sum = 0.0f;
    for (int d = 0; d < d_k; d++) {
        sum += Q[q_offset + d] * K[k_offset + d];
    }
    
    O[idx] = sum * scale;
}

extern "C" void qk_mul_launcher(
    const float* Q,
    const float* K,
    float* O,
    int batch_size,
    int num_heads,
    int seq_len,
    int d_k,
    float scale) {
    
    int total_elements = batch_size * num_heads * seq_len * seq_len;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    // defoult stream (0)
    qk_mul_kernel<<<grid_size, block_size, 0, 0>>>(
        Q, K, O, batch_size, num_heads, seq_len, d_k, scale
    );
}