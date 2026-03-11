#include <cuda_fp16.h>
#include <stdio.h>

extern __global__ void flash_attn_bc16(
    const __half* __restrict__,
    const __half* __restrict__,
    const __half* __restrict__,
    __half* __restrict__,
    int, int, float);

extern __global__ void flash_attn_bc32(
    const __half* __restrict__,
    const __half* __restrict__,
    const __half* __restrict__,
    __half* __restrict__,
    int, int, float);

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
    int Br,
    int Bc
) {

    dim3 block_size(32, Br / (16 / (Bc / 16)));

    int blocks = batch_size * num_heads * ((seq_len + Br - 1) / Br);
    size_t shared_mem_size;

    switch (Bc)
    {
        case 16:
            shared_mem_size =
            (Br * d_k * sizeof(float)) +        // s_O
            (Br * (d_k + 8) * sizeof(__half)) + // s_Q
            (Bc * (d_k + 8) * sizeof(__half)) + // s_K
            ((Bc + 2) * d_k * sizeof(__half)) + // s_V
            (Br * sizeof(float)) * 2;           // s_m_prev, s_d_prev

            flash_attn_bc16<<<blocks, block_size, shared_mem_size>>>(
                Q,K,V,O,seq_len,d_k,scale
            );
            break;

        case 32:
            shared_mem_size =
            (Br * d_k * sizeof(float)) +         // s_O
            (2 * Br * sizeof(float)) * 2 +       // s_row_max, s_row_sum
            ((Bc + 8) * Br * sizeof(__half)) +   // s_A
            (Br * (d_k + 8) * sizeof(__half)) +  // s_Q
            (Bc * (d_k + 8) * sizeof(__half)) +  // s_K
            ((Bc + 8) * d_k * sizeof(__half)) +  // s_V
            (Br * sizeof(float)) * 2;            // s_m_prev, s_d_prev

            cudaFuncSetAttribute(
                flash_attn_bc32,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                shared_mem_size
            );

            flash_attn_bc32<<<blocks, block_size, shared_mem_size>>>(
                Q,K,V,O,seq_len,d_k,scale
            );
            break;

        default:
            printf("Unsupported Bc\n");
    }
}