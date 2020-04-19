#pragma once
#include <initializer_list>
#include <ostream>
#include "tensor_gpu.cuh"
#include <vector>

class __declspec(dllexport) tensor
{
	friend class tensor_initializer;
public:
	enum INIT
	{
		NONE,
		ZERO,
		VALUE
	};

	tensor();
	tensor(std::initializer_list<int> p_shape, INIT p_init = ZERO, float p_value = 0.f);
	tensor(std::initializer_list<int> p_shape, float* p_data);
	tensor(const tensor& p_copy);
	tensor& operator=(const tensor& p_copy);
	~tensor();

	static tensor zero(std::initializer_list<int> p_shape);
	static tensor zero_like(const tensor& p_copy);
	static tensor value(std::initializer_list<int> p_shape, float p_value);
	static tensor value_like(tensor& p_copy, float p_value);

	// properties
	int rank() const { return _rank; }
	int shape(const int p_index) const { return _shape[p_index]; }
	int size() const { return _size; }
	float* data() const { return _data; }

	void fill(float p_value) const;
	void reshape(std::initializer_list<int> p_new_shape);
	void resize(std::initializer_list<int> p_shape, INIT p_init = ZERO, float p_value = 0.f);
	void resize(int p_rank, int* p_shape, INIT p_init = ZERO, float p_value = 0.f);
	void override(tensor& p_copy);
	static void concat(std::vector<tensor*> &p_source, tensor& p_dest, int p_dim);
	static void split(tensor& p_source, std::vector<tensor*> &p_dest);

	tensor mean(int p_dim) const;

	// arithmetic operators
	tensor& operator += (const tensor& p_rhs);
	tensor& operator += (float p_rhs);
	tensor& operator -= (const tensor& p_rhs);
	tensor& operator -= (float p_rhs);
	tensor& operator *= (const tensor& p_rhs);
	tensor& operator *= (float p_rhs);
	tensor& operator /= (float p_rhs);
	tensor operator + (const tensor& p_rhs) const;
	tensor operator + (float p_rhs) const;
	friend tensor operator + (float p_lhs, const tensor& p_rhs);
	tensor operator - (const tensor& p_rhs) const;
	tensor operator - (float p_rhs) const;
	friend tensor operator - (float p_lhs, const tensor& p_rhs);
	tensor operator * (const tensor& p_rhs) const;
	tensor operator * (float p_rhs) const;
	friend tensor operator * (float p_lhs, const tensor& p_rhs);
	tensor operator / (float p_rhs) const;
	friend tensor operator / (float p_lhs, const tensor& p_rhs);

	// operators
	float& operator[] (int p_index) const;
	void T();
	std::vector<int> max_index(int p_dim = 0) const;
	tensor gather(std::vector<int> &p_index) const;
	float max() const;
	float min() const;

	//gpu operations
	void to_gpu();
	void to_cpu();

	friend std::ostream &operator<<(std::ostream &output, const tensor &p_tensor) {
		if (p_tensor.rank() == 1)
		{
			print_vector(output, p_tensor);
		}
		if (p_tensor.rank() == 2)
		{
			print_matrix(output, p_tensor);
		}
		if (p_tensor.rank() == 3)
		{
			print_volume(output, p_tensor);
		}
		if (p_tensor.rank() == 4)
		{
			print_batch_volume(output, p_tensor);
		}
		return output;
	}

private:
	tensor(int p_rank, int* p_shape, INIT p_init = ZERO, float p_value = 0.f);

	static void print_vector(std::ostream &output, const tensor &p_tensor);
	static void print_matrix(std::ostream &output, const tensor &p_tensor);
	static void print_volume(std::ostream &output, const tensor &p_tensor);
	static void print_batch_volume(std::ostream &output, const tensor &p_tensor);

	static void check_gpu(const tensor& p_lhs, const tensor& p_rhs);
	int check_shape(std::initializer_list<int> &p_shape) const;
	int check_shape(int p_rank, const int* p_shape) const;

	static int* init_shape(int &p_rank, std::initializer_list<int> &p_shape);
	static int* init_shape(int &p_rank, int* p_shape);
	static int	init_size(int &p_rank, const int* p_shape);
	static int*	init_stride(int &p_rank, const int* p_shape);
	static float* init_data(int &p_size, INIT p_init, float p_value);

	static void fill(float* p_data, int p_size, INIT p_init, float p_value = 0.f);

	float* gpu_data() const;

	int		_rank;
	int		_size;
	int*	_shape;
	int*	_stride;
	float*	_data;

	tensor_gpu* _gpu_data;

	bool _gpu_flag;
	bool _transpose_flag;

	int	_end;

	const int SHAPE_EQUAL = 0;
	const int SHAPE_EQUAL_DIFF_SIZE = 1;
	const int SHAPE_DIFF = 2;
};

