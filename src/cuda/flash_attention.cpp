#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

extern "C" void flash_attention_launcher(
    const __half* Q, const __half* K, const __half* V, __half* O,
    int batch_size, int num_heads, int seq_len, int d_k,
    float scale, int Br);

torch::Tensor flash_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    float scale = 1.0f,
    int Br = 32) {
    
    // Проверки
    TORCH_CHECK(query.is_cuda(), "Query must be on CUDA");
    TORCH_CHECK(key.is_cuda(), "Key must be on CUDA");
    TORCH_CHECK(value.is_cuda(), "Value must be on CUDA");
    
    TORCH_CHECK(query.sizes() == key.sizes(), "Q and K must have same shape");
    TORCH_CHECK(query.size(0) == value.size(0), "Batch size mismatch");
    TORCH_CHECK(query.size(1) == value.size(1), "Head count mismatch");
    TORCH_CHECK(query.size(3) == value.size(3), "Head dimension mismatch");
    
    TORCH_CHECK(query.dtype() == torch::kFloat16, "Only float16 supported for this FlashAttention");
    
    auto sizes = query.sizes();
    int batch_size = sizes[0];
    int num_heads = sizes[1];
    int seq_len = sizes[2];
    int d_k = sizes[3];
    
    // Reshape tensors to 2D for kernel: (batch*head*seq_len, d_k)
    auto query_flat = query.view({-1, d_k});
    auto key_flat = key.view({-1, d_k});
    auto value_flat = value.view({-1, d_k});
    
    // Create output tensor
    auto output = torch::empty(
        {batch_size * num_heads * seq_len, d_k}, 
        torch::TensorOptions()
            .dtype(torch::kFloat16)
            .device(torch::kCUDA)
            .requires_grad(false));
    
    // Apply scaling factor (scale / sqrt(d_k))
    float actual_scale = scale / sqrtf(static_cast<float>(d_k));
    
    // Launch kernel with reshaped tensors
    flash_attention_launcher(
        reinterpret_cast<const __half*>(query_flat.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(key_flat.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(value_flat.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        batch_size, num_heads, seq_len, d_k,
        actual_scale,
        Br
    );
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA error in FlashAttention: ", cudaGetErrorString(err));
    }
    
    // Reshape output back to original shape
    return output.view({batch_size, num_heads, seq_len, d_k});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention_forward,
      "Flash Attention forward pass (CUDA)",
      py::arg("query"), py::arg("key"), py::arg("value"),
      py::arg("scale") = 1.0f,
      py::arg("Br") = 32);
}