#pragma once

class __declspec(dllexport) tensor_operator_cpu
{
public:
	static void add(float* p_x, int p_x_size, float* p_y, int p_y_size, float* p_z);
	static void const_add(float* p_x, float p_y, float* p_z, int p_size);
	static void sub(float* p_x, int p_x_size, float* p_y, int p_y_size, float* p_z);
	static void const_sub(float* p_x, float p_y, float* p_z, int p_size);
	static void const_sub(float p_x, float* p_y, float* p_z, int p_size);
	static void mul(float* p_x, bool p_transpose_x, float* p_y, bool p_transpose_y, float* p_z, int p_rows, int p_common, int p_cols);
	static void const_mul(float* p_x, float p_y, float* p_z, int p_size);
	static void const_div(float p_x, float* p_y, float* p_z, int p_size);
	static void reduce_sum(float* p_x, int p_x_shape, float* p_y, int p_y_size);

private:
	static void mul_ab(float* p_x, float* p_y, float* p_z, int p_rows, int p_common, int p_cols);
	static void mul_aTb(float* p_x, float* p_y, float* p_z, int p_rows, int p_common, int p_cols);
	static void mul_abT(float* p_x, float* p_y, float* p_z, int p_rows, int p_common, int p_cols);
	static void add_broadcast_x(float* p_x, int p_x_size, float* p_y, int p_y_size, float* p_z);
	static void add_broadcast_y(float* p_x, int p_x_size, float* p_y, int p_y_size, float* p_z);
	static void sub_broadcast_x(float* p_x, int p_x_size, float* p_y, int p_y_size, float* p_z);
	static void sub_broadcast_y(float* p_x, int p_x_size, float* p_y, int p_y_size, float* p_z);


	tensor_operator_cpu();
	~tensor_operator_cpu();
};

