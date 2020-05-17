#include "tensor_operator_gpu.cuh"
#include "cuda_kernels.cuh"
#include <cmath>
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cublas_v2.h>

void tensor_operator_gpu::add(float* p_x, float* p_y, float* p_z, const int p_size)
{
	call_kernel(add_kernel, p_x, p_y, p_z, p_size);
}

void tensor_operator_gpu::const_add(float* p_x, const float p_y, float* p_z, const int p_size)
{
	call_kernel(const_add_kernel, p_x, p_y, p_z, p_size);
}

void tensor_operator_gpu::sub(float* p_x, float* p_y, float* p_z, const int p_size)
{
	call_kernel(sub_kernel, p_x, p_y, p_z, p_size);
}

void tensor_operator_gpu::const_sub(float* p_x, const float p_y, float* p_z, const int p_size)
{
	call_kernel(const_sub_kernel, p_x, p_y, p_z, p_size);
}

void tensor_operator_gpu::const_sub(const float p_x, float* p_y, float* p_z, const int p_size)
{
	call_kernel(const_sub_kernel, p_x, p_y, p_z, p_size);
}

void tensor_operator_gpu::mul(float* p_x, bool p_transpose_x, float* p_y, bool p_transpose_y, float* p_z, int p_rows, int p_common, int p_cols)
{
	cublasHandle_t handle;
	cublasCreate(&handle);

	const cublasOperation_t transpose_x = p_transpose_x ? CUBLAS_OP_T : CUBLAS_OP_N;
	const cublasOperation_t transpose_y = p_transpose_y ? CUBLAS_OP_T : CUBLAS_OP_N;
	const int lda = p_transpose_x ? p_rows : p_common;
	const int ldb = p_transpose_y ? p_common : p_cols;

	float alpha = 1.f;
	float beta = 0.f;


	cublasSgemm(handle,
		transpose_x, transpose_y,
		p_rows, p_common, p_cols,
		&alpha, // 1
		p_x, lda,
		p_y, ldb,
		&beta, // 0
		p_z, p_cols);

	cudaDeviceSynchronize();
}

void tensor_operator_gpu::const_mul(float* p_x, const float p_y, float* p_z, const int p_size)
{
	call_kernel(const_mul_kernel, p_x, p_y, p_z, p_size);
}

void tensor_operator_gpu::const_div(const float p_x, float* p_y, float* p_z, const int p_size)
{
	call_kernel(const_div_kernel, p_x, p_y, p_z, p_size);
}

void tensor_operator_gpu::call_kernel(void(* kernel)(const float*, const float*, float*, int), float* p_x, float* p_y, float* p_z, const int p_size)
{
	const int blockSize = 256;
	int gridSize = static_cast<int>(ceil(static_cast<float>(p_size) / blockSize));

	kernel<<<gridSize, blockSize >> >(p_x, p_y, p_z, p_size);
	const cudaError_t cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
	}
	cudaError_t error = cudaDeviceSynchronize();
}

void tensor_operator_gpu::call_kernel(void(* kernel)(const float*, const float, float*, const int), float* p_x, const float p_y, float* p_z, const int p_size)
{
	const int blockSize = 256;
	int gridSize = static_cast<int>(ceil(static_cast<float>(p_size) / blockSize));

	kernel <<<gridSize, blockSize >> >(p_x, p_y, p_z, p_size);
	const cudaError_t cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
	}
	cudaError_t error = cudaDeviceSynchronize();
}

void tensor_operator_gpu::call_kernel(void(* kernel)(const float, const float*, float*, const int), const float p_x, float* p_y, float* p_z, const int p_size)
{
	const int blockSize = 256;
	int gridSize = static_cast<int>(ceil(static_cast<float>(p_size) / blockSize));

	kernel <<<gridSize, blockSize >> >(p_x, p_y, p_z, p_size);
	const cudaError_t cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
	}
	cudaError_t error = cudaDeviceSynchronize();
}

tensor_operator_gpu::tensor_operator_gpu()
= default;


tensor_operator_gpu::~tensor_operator_gpu()
= default;

