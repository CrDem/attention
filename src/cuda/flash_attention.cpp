#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

extern "C" void qk_mul_launcher(
    const float* Q, const float* K, float* O,
    int batch_size, int num_heads, int seq_len, int d_k,
    float scale);

torch::Tensor flash_attention_qk_mul(
    torch::Tensor query,
    torch::Tensor key,
    float scale = 1.0f) {
    
    TORCH_CHECK(query.is_cuda(), "Query must be on CUDA");
    TORCH_CHECK(key.is_cuda(), "Key must be on CUDA");
    TORCH_CHECK(query.sizes() == key.sizes(), "Q and K must have same shape");
    TORCH_CHECK(query.dtype() == torch::kFloat32, "Only float32 supported");
    
    auto sizes = query.sizes();
    int batch_size = sizes[0];
    int num_heads = sizes[1];
    int seq_len = sizes[2];
    int d_k = sizes[3];
    
    auto output = torch::empty(
        {batch_size, num_heads, seq_len, seq_len}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    qk_mul_launcher(
        query.data_ptr<float>(),
        key.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, num_heads, seq_len, d_k,
        scale
    );
    
    // Проверяем ошибки
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA error: ", cudaGetErrorString(err));
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qk_mul", &flash_attention_qk_mul, 
          "Flash Attention Q*K multiplication (CUDA)",
          py::arg("query"), py::arg("key"), py::arg("scale") = 1.0f);
}