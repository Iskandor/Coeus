#include "device_launch_parameters.h"

__global__ void add_kernel(const float* p_x, const float* p_y, float* p_z, const int p_size)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < p_size)
	{
		p_z[i] = p_x[i] + p_y[i];
	}
}

__global__ void const_add_kernel(const float* p_x, const float p_y, float* p_z, const int p_size)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < p_size)
	{
		p_z[i] = p_x[i] + p_y;
	}
}

__global__ void sub_kernel(const float* p_x, const float* p_y, float* p_z, const int p_size)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < p_size)
	{
		p_z[i] = p_x[i] - p_y[i];
	}
}

__global__ void const_sub_kernel(const float* p_x, const float p_y, float* p_z, const int p_size)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < p_size)
	{
		p_z[i] = p_x[i] - p_y;
	}
}

__global__ void const_sub_kernel(const float p_x, const float* p_y, float* p_z, const int p_size)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < p_size)
	{
		p_z[i] = p_x - p_y[i];
	}
}

__global__ void const_mul_kernel(const float* p_x, const float p_y, float* p_z, const int p_size)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < p_size)
	{
		p_z[i] = p_x[i] * p_y;
	}
}

__global__ void const_div_kernel(const float p_x, const float* p_y, float* p_z, const int p_size)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < p_size)
	{
		p_z[i] = p_x / p_y[i];
	}
}