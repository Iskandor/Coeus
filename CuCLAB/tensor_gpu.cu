#include "tensor_gpu.cuh"
#include <cuda_runtime.h>


tensor_gpu::tensor_gpu(const int p_size) : _size(p_size)
{
	_data = init_data(_size);
}

tensor_gpu::tensor_gpu(const tensor_gpu& p_copy)
{
	_size = p_copy._size;
	_data = init_data(_size);
	cudaMemcpy(_data, p_copy._data, sizeof(float) * p_copy._size, cudaMemcpyDeviceToDevice);
}

tensor_gpu& tensor_gpu::operator=(const tensor_gpu& p_copy)
{
	if (_size != p_copy._size)
	{
		cudaFree(_data);
		_data = init_data(_size);
	}
	cudaMemcpy(_data, p_copy._data, sizeof(float) * p_copy._size, cudaMemcpyDeviceToDevice);
	_size = p_copy._size;

	return *this;
}

tensor_gpu::~tensor_gpu()
{
	_size = 0;
	cudaFree(_data);
}

void tensor_gpu::to_gpu(float* p_cpu_data) const
{
	cudaError_t error = cudaMemcpy(_data, p_cpu_data, sizeof(float) * _size, cudaMemcpyHostToDevice);
}

void tensor_gpu::to_cpu(float* p_cpu_data) const
{
	cudaError_t error = cudaMemcpy(p_cpu_data, _data, sizeof(float) * _size, cudaMemcpyDeviceToHost);
}

float* tensor_gpu::init_data(int& p_size)
{
	float* result;
	cudaError_t error = cudaMalloc(reinterpret_cast<void**>(&result), sizeof(float) * p_size);
	return result;
}