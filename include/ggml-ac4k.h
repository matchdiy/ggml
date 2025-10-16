#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_ac4k_init(int device, const char* name);

GGML_BACKEND_API bool ggml_backend_is_ac4k(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_ac4k_buffer_type(int device);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_ac4k_split_buffer_type(int main_device, const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_ac4k_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_ac4k_get_device_count(void);
GGML_BACKEND_API void ggml_backend_ac4k_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_ac4k_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_ac4k_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_ac4k_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_ac4k_reg(void);

#ifdef  __cplusplus
}
#endif
