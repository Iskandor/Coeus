#pragma once

class __declspec(dllexport) tensor_operator_gpu
{
public:
	static void add(float* p_x, float* p_y, float* p_z, int p_size);
	static void const_add(float* p_x, float p_y, float* p_z, int p_size);
	static void sub(float* p_x, float* p_y, float* p_z, int p_size);
	static void const_sub(float* p_x, float p_y, float* p_z, int p_size);
	static void const_sub(float p_x, float* p_y, float* p_z, int p_size);
	static void mul(float* p_x, bool p_transpose_x, float* p_y, bool p_transpose_y, float* p_z, int p_rows, int p_common, int p_cols);
	static void const_mul(float* p_x, float p_y, float* p_z, int p_size);
	static void const_div(float p_x, float* p_y, float* p_z, int p_size);

private:
	static void call_kernel(void(*kernel)(const float*, const float*, float*, const int), float* p_x, float* p_y, float* p_z, int p_size);
	static void call_kernel(void(*kernel)(const float*, const float, float*, const int), float* p_x, float p_y, float* p_z, int p_size);
	static void call_kernel(void(*kernel)(const float, const float*, float*, const int), float p_x, float* p_y, float* p_z, int p_size);

	tensor_operator_gpu();
	~tensor_operator_gpu();
};