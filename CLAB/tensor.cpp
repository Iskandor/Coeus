#include "tensor.h"
#include <cstring>
#include <cstdlib>
#include <cassert>
#include "tensor_operator_cpu.h"
#include "tensor_operator_gpu.cuh"


tensor::tensor() :
	_rank(0),
	_size(0),
	_shape(nullptr),
	_stride(nullptr),
	_data(nullptr),
	_gpu_data(nullptr),
	_gpu_flag(false),
	_transpose_flag(false),
	_end(0)
{
}

tensor::tensor(std::initializer_list<int> p_shape, const INIT p_init, const float p_value): 
	_gpu_data(nullptr),
	_gpu_flag(false),
	_transpose_flag(false)
{
	_rank = static_cast<int>(p_shape.size());
	_shape = init_shape(_rank, p_shape);
	_size = init_size(_rank, _shape);
	_stride = init_stride(_rank, _shape);
	_data = init_data(_size, p_init, p_value);
	_end = 0;
}

tensor::tensor(std::initializer_list<int> p_shape, float* p_data) :
	_gpu_data(nullptr),
	_gpu_flag(false),
	_transpose_flag(false)
{
	_rank = static_cast<int>(p_shape.size());
	_shape = init_shape(_rank, p_shape);
	_size = init_size(_rank, _shape);
	_stride = init_stride(_rank, _shape);
	_data = init_data(_size, ZERO, 0.f);
	memcpy(_data, p_data, sizeof(float) * _size);
	_end = 0;
}

tensor::tensor(const tensor& p_copy)
{
	_rank = p_copy._rank;
	_size = p_copy._size;
	_shape = static_cast<int*>(malloc(sizeof(int) * p_copy._rank));
	memcpy(_shape, p_copy._shape, sizeof(int) * p_copy._rank);
	_stride = static_cast<int*>(malloc(sizeof(int) * p_copy._rank));
	memcpy(_stride, p_copy._stride, sizeof(int) * p_copy._rank);
	_data = static_cast<float*>(malloc(sizeof(float) * p_copy._size));
	memcpy(_data, p_copy._data, sizeof(float) * p_copy._size);
	_end = 0;

	if (p_copy._gpu_flag)
	{
		_gpu_data = new tensor_gpu(p_copy._size);
		*_gpu_data = *p_copy._gpu_data;
	}
	else
	{
		_gpu_data = nullptr;
	}
	_gpu_flag = p_copy._gpu_flag;
	_transpose_flag = p_copy._transpose_flag;
}

tensor& tensor::operator=(const tensor& p_copy)
{
	if (_rank != p_copy._rank)
	{
		free(_shape);
		_shape = static_cast<int*>(malloc(sizeof(int) * p_copy._rank));
		free(_stride);
		_stride = static_cast<int*>(malloc(sizeof(int) * p_copy._rank));
	}
	if (_size != p_copy._size)
	{
		free(_data);
		_data = static_cast<float*>(malloc(sizeof(float) * p_copy._size));
		_end = 0;
	}
	memcpy(_shape, p_copy._shape, sizeof(int) * p_copy._rank);
	memcpy(_stride, p_copy._stride, sizeof(int) * p_copy._rank);
	memcpy(_data, p_copy._data, sizeof(float) * p_copy._size);

	_rank = p_copy._rank;
	_size = p_copy._size;

	if (p_copy._gpu_flag)
	{
		if (!_gpu_flag) _gpu_data = new tensor_gpu(p_copy._size);
		*_gpu_data = *p_copy._gpu_data;
	}
	else
	{
		_gpu_data = nullptr;
	}
	_gpu_flag = p_copy._gpu_flag;
	_transpose_flag = p_copy._transpose_flag;

	return *this;
}


tensor::~tensor()
{
	_rank = 0;
	_size = 0;
	delete _shape;
	_shape = nullptr;
	delete _stride;
	_stride = nullptr;
	delete _data;
	_data = nullptr;
	delete _gpu_data;
	_gpu_data = nullptr;
	_end = 0;
}

tensor tensor::zero(const std::initializer_list<int> p_shape)
{
	return tensor(p_shape, ZERO, 0.f);
}

tensor tensor::zero_like(const tensor& p_copy)
{
	return tensor(p_copy._rank, p_copy._shape);
}

tensor tensor::value(const std::initializer_list<int> p_shape, const float p_value)
{
	return tensor(p_shape, VALUE, p_value);
}

tensor tensor::value_like(tensor& p_copy, float p_value)
{
	return tensor(p_copy._rank, p_copy._shape);
}

void tensor::fill(const float p_value) const
{
	fill(_data, _size, VALUE, p_value);
}

void tensor::reshape(std::initializer_list<int> p_new_shape)
{
	int rank = p_new_shape.size();
	int* shape = init_shape(rank, p_new_shape);
	int size = init_size(rank, shape);

	if (_size != size)
	{
		assert(0);
	}
	else
	{
		_rank = rank;
		delete _shape;
		_shape = shape;
	}
}

void tensor::resize(std::initializer_list<int> p_shape, const INIT p_init, const float p_value)
{
	const int shape_check = check_shape(p_shape);
	if (shape_check == SHAPE_DIFF)
	{
		_rank = p_shape.size();
	}
	if (shape_check == SHAPE_DIFF || shape_check == SHAPE_EQUAL_DIFF_SIZE)
	{
		free(_shape);
		_shape = init_shape(_rank, p_shape);
		free(_stride);
		_stride = init_stride(_rank, _shape);
		_size = init_size(_rank, _shape);
		free(_data);
		_data = init_data(_size, p_init, p_value);
		_end = 0;

		if (_gpu_flag)
		{
			delete _gpu_data;
		}
	}
	if (shape_check == SHAPE_EQUAL)
	{
		fill(_data, _size, p_init, p_value);
	}

	_gpu_flag = false;
	_transpose_flag = false;
}

void tensor::resize(const int p_rank, int* p_shape, const INIT p_init, const float p_value)
{
	const int shape_check = check_shape(p_rank, p_shape);
	if (shape_check == SHAPE_DIFF)
	{
		_rank = p_rank;
	}
	if (shape_check == SHAPE_DIFF || shape_check == SHAPE_EQUAL_DIFF_SIZE)
	{
		free(_shape);
		_shape = init_shape(_rank, p_shape);
		free(_stride);
		_stride = init_stride(_rank, _shape);
		_size = init_size(_rank, _shape);
		free(_data);
		_data = init_data(_size, p_init, p_value);
		_end = 0;

		if (_gpu_flag)
		{
			delete _gpu_data;
		}
	}

	_gpu_flag = false;
	_transpose_flag = false;
}


void tensor::override(tensor& p_copy)
{
	if (_size == p_copy._size)
	{
		memcpy(_data, p_copy._data, sizeof(float) * p_copy._size);
	}
	else
	{
		resize(p_copy._rank, p_copy._shape, ZERO, 0.f);
		memcpy(_data, p_copy._data, sizeof(float) * p_copy._size);
	}
}

void tensor::concat(std::vector<tensor*> &p_source, tensor& p_dest, int p_dim)
{
	int rank = 0;
	for (auto t : p_source)
	{
		rank = t->_rank;
	}
	
	if (rank == 1)
	{
		if (p_dim == 0)
		{
			int size = 0;
			for (auto t : p_source)
			{
				size += t->_size;
			}
			p_dest.resize({ size });
			int index = 0;
			for (auto t : p_source)
			{
				memcpy(p_dest._data + index, t->_data, sizeof(float) * t->_size);
				index += t->_size;
			}
		}
		if (p_dim == 1)
		{
			int rows = 0;
			int cols = 0;

			for (auto t : p_source)
			{
				rows++;
				cols = t->_size;
			}

			p_dest.resize({ rows, cols });

			int index = 0;
			for (auto t : p_source)
			{
				memcpy(p_dest._data + index, t->_data, sizeof(float) * t->_size);
				index += t->_size;
			}
		}
	}
	if (rank == 2)
	{
		int rows = 0;
		int cols = 0;

		if (p_dim == 0)
		{
			for (auto t : p_source)
			{
				rows = t->_shape[0];
				cols += t->_shape[1];
			}
			p_dest.resize({ rows, cols });

			int index = 0;
			for (int i = 0; i < p_dest._shape[0]; i++)
			{
				for (auto t : p_source)
				{
					memcpy(p_dest._data + index, t->_data + i * t->_shape[1], sizeof(float) * t->_shape[1]);
					index += t->_shape[1];
				}
			}
		}
		if (p_dim == 1)
		{
			for (auto t : p_source)
			{
				rows += t->_shape[0];
				cols = t->_shape[1];
			}
			p_dest.resize({ rows, cols });

			int index = 0;
			for (auto t : p_source)
			{
				memcpy(p_dest._data + index, t->_data, sizeof(float) * t->_size);
				index += t->_size;
			}
		}
	}
}

void tensor::split(tensor& p_source, std::vector<tensor*>& p_dest)
{
	float* sx = p_source._data;

	if (p_source._rank == 1)
	{
	}

	if (p_source._rank == 2)
	{
		for (auto& dest : p_dest)
		{
			dest->resize({ p_source._shape[0], dest->_shape[1] });
		}

		for (int i = 0; i < p_source._shape[0]; i++)
		{
			for (const auto& dest : p_dest)
			{
				memcpy(dest->_data + i * dest->_shape[1], sx, sizeof(float) * dest->_shape[1]);
				sx += dest->_shape[1];
			}
		}
	}
}

tensor tensor::mean(int p_dim) const
{
	tensor result;

	if (_rank == 1)
	{
		result.resize({ 1 });

		float* x = _data;
		for (int i = 0; i < _shape[0]; i++)
		{
			result[i] += *x++;
		}

		result /= _shape[0];
	}

	if (_rank == 2)
	{
		if (p_dim == 0)
		{
			result.resize({ _shape[0], 1 });

			float* x = _data;

			for(int i = 0; i < _shape[0]; i++)
			{
				for (int j = 0; j < _shape[1]; j++)
				{
					result[i] += *x++;
				}
			}

			result /= _shape[1];
		}
		if (p_dim == 1)
		{
			result.resize({ 1, _shape[1] });

			float* x = _data;

			for (int i = 0; i < _shape[0]; i++)
			{
				for (int j = 0; j < _shape[1]; j++)
				{
					result[j] += *x++;
				}
			}

			result /= _shape[0];
		}
	}

	return result;
}

tensor& tensor::operator+=(const tensor& p_rhs)
{
	check_gpu(*this, p_rhs);
	if (_gpu_flag)
	{
		tensor_operator_gpu::add(_gpu_data->_data, p_rhs._gpu_data->_data, _gpu_data->_data, _size);
	}
	else
	{
		tensor_operator_cpu::add(_data, _size, p_rhs._data, p_rhs._size, _data);
	}	
	return *this;
}

tensor& tensor::operator+=(const float p_rhs)
{
	if (_gpu_flag)
	{
		tensor_operator_gpu::const_add(_gpu_data->_data, p_rhs, _gpu_data->_data, _size);
	}
	else
	{
		tensor_operator_cpu::const_add(_data, p_rhs, _data, _size);
	}
	return *this;
}

tensor& tensor::operator-=(const tensor& p_rhs)
{	
	check_gpu(*this, p_rhs);
	if (_gpu_flag)
	{
		tensor_operator_gpu::sub(_gpu_data->_data, p_rhs._gpu_data->_data, _gpu_data->_data, _size);
	}
	else
	{
		tensor_operator_cpu::sub(_data, _size, p_rhs._data, p_rhs._size, _data);
	}
	return *this;
}

tensor& tensor::operator-=(const float p_rhs)
{
	if (_gpu_flag)
	{
		tensor_operator_gpu::const_sub(_gpu_data->_data, p_rhs, _gpu_data->_data, _size);
	}
	else
	{
		tensor_operator_cpu::const_sub(_data, p_rhs, _data, _size);
	}
	return *this;
}

tensor& tensor::operator*=(const tensor& p_rhs)
{
	check_gpu(*this, p_rhs);

	if (_gpu_flag)
	{
		tensor_operator_gpu::mul(_gpu_data->_data, _transpose_flag, p_rhs._gpu_data->_data, p_rhs._transpose_flag, _gpu_data->_data, _shape[0], p_rhs._shape[0], p_rhs._shape[1]);
	}
	else
	{
		tensor_operator_cpu::mul(_data, _transpose_flag, p_rhs._data, p_rhs._transpose_flag, _data, _shape[0], p_rhs._shape[0], p_rhs._shape[1]);
	}
	return *this;
}

tensor& tensor::operator*=(const float p_rhs)
{
	if (_gpu_flag)
	{
		tensor_operator_gpu::const_mul(_gpu_data->_data, p_rhs, _gpu_data->_data, _size);
	}
	else
	{
		tensor_operator_cpu::const_mul(_data, p_rhs, _data, _size);
	}
	return *this;
}

tensor& tensor::operator/=(const float p_rhs)
{
	if (_gpu_flag)
	{
		tensor_operator_gpu::const_mul(_gpu_data->_data, 1.f / p_rhs, _gpu_data->_data, _size);
	}
	else
	{
		tensor_operator_cpu::const_mul(_data, 1.f / p_rhs, _data, _size);
	}
	return *this;
}

tensor tensor::operator+(const tensor& p_rhs) const
{
	tensor result = _size >= p_rhs._size ? *this : p_rhs;
	return _size >= p_rhs._size ? result += p_rhs : result += *this;
}

tensor tensor::operator+(const float p_rhs) const
{
	tensor result = *this;
	return result += p_rhs;
}

tensor tensor::operator-(const tensor& p_rhs) const
{
	tensor result = _size >= p_rhs._size ? *this : p_rhs;
	return _size >= p_rhs._size ? result -= p_rhs : result -= *this;
}

tensor tensor::operator-(const float p_rhs) const
{
	tensor result = *this;
	return result -= p_rhs;
}

tensor tensor::operator*(const tensor& p_rhs) const
{
	check_gpu(*this, p_rhs);

	const int rows = _transpose_flag ? _shape[1] : _shape[0];
	const int cols = p_rhs._transpose_flag ? p_rhs._shape[0] : p_rhs._shape[1];
	const int common = _transpose_flag ? _shape[0] : _shape[1];

	tensor result({ rows, cols });

	if (_gpu_flag)
	{
		tensor_operator_gpu::mul(_gpu_data->_data, _transpose_flag, p_rhs._gpu_data->_data, p_rhs._transpose_flag, result._gpu_data->_data, rows, common, cols);
	}
	else
	{
		tensor_operator_cpu::mul(_data, _transpose_flag, p_rhs._data, p_rhs._transpose_flag, result._data, rows, common, cols);
	}
	return result;
}

tensor tensor::operator*(const float p_rhs) const
{
	tensor result = *this;
	return result *= p_rhs;
}

tensor tensor::operator/(const float p_rhs) const
{
	tensor result = *this;
	return result /= p_rhs;
}

float& tensor::operator[](const int p_index) const
{
	return _data[p_index];
}

void tensor::T()
{
	_transpose_flag = !_transpose_flag;
}

std::vector<int> tensor::max_index(const int p_dim) const
{
	std::vector<int> result;

	if (_rank == 1)
	{
		result.resize(1);
		result[0] = 0;
		for(int i = 1; i < _size; i++)
		{
			if (_data[result[0]] < _data[i])
			{
				result[0] = i;
			}
		}		
	}
	if (_rank == 2)
	{
		if (p_dim == 0)
		{
			result.resize(_shape[0]);

			for (int i = 0; i < _shape[0]; i++)
			{
				result[i] = i * _shape[1];
			}

			for (int i = 0; i < _shape[0]; i++)
			{
				for (int j = 1; j < _shape[1]; j++)
				{
					if (_data[result[i]] < _data[i * _shape[1] + j])
					{
						result[i] = i * _shape[1] + j;
					}
				}
			}
		}
		if (p_dim == 1)
		{
			result.resize(_shape[1]);

			for (int j = 0; j < _shape[1]; j++)
			{
				result[j] = j;
			}

			for (int i = 1; i < _shape[0]; i++)
			{
				for (int j = 0; j < _shape[1]; j++)
				{
					if (_data[result[j]] < _data[i * _shape[1] + j])
					{
						result[j] = i * _shape[1] + j;
					}
				}
			}
		}
	}

	return result;
}

tensor tensor::gather(std::vector<int> &p_index) const
{
	tensor result({_shape[0]});

	float *rx = result._data;

	for(int i = 0; i < _shape[0]; i++)
	{
		*rx++ = _data[p_index[i]];
	}

	return result;
}

float tensor::max() const
{
	float max = _data[0];
	float *x = &_data[1];

	for(int i = 1; i < _size; i++)
	{
		if (max < *x)
		{
			max = *x;
		}
		x++;
	}

	return max;
}

float tensor::min() const
{
	float min = _data[0];
	float *x = &_data[1];

	for (int i = 1; i < _size; i++)
	{
		if (min > *x)
		{
			min = *x;
		}
		x++;
	}

	return min;
}

void tensor::to_gpu()
{
	if (_gpu_data == nullptr)
	{
		_gpu_data = new tensor_gpu(_size);
	}
	_gpu_data->to_gpu(_data);
	_gpu_flag = true;
}

void tensor::to_cpu()
{
	_gpu_data->to_cpu(_data);
	_gpu_flag = false;
}

tensor::tensor(const int p_rank, int* p_shape, const INIT p_init, const float p_value)
{
	_rank = p_rank;
	_shape = static_cast<int*>(malloc(sizeof(int) * p_rank));
	memcpy(_shape, p_shape, sizeof(int) * p_rank);
	_stride = init_stride(_rank, _shape);
	_size = init_size(_rank, _shape);
	_data = init_data(_size, p_init, p_value);
	_end = 0;
	_gpu_flag = false;
	_gpu_data = nullptr;
	_transpose_flag = false;
	
}

void tensor::print_vector(std::ostream& output, const tensor& p_tensor)
{
	for (int i = 0; i < p_tensor._size; i++) {
		if (i == p_tensor._size - 1) {
			output << p_tensor._data[i];
		}
		else {
			output << p_tensor._data[i] << ",";
		}
	}
}

void tensor::print_matrix(std::ostream& output, const tensor& p_tensor)
{
	float* x = p_tensor._data;
	for (int i = 0; i < p_tensor._shape[0]; i++) {
		for (int j = 0; j < p_tensor._shape[1]; j++)
		{
			output << *x++;
			if (j < p_tensor._shape[1] - 1) {
				output << ",";
			}
		}
		output << std::endl;
	}
}

void tensor::print_volume(std::ostream& output, const tensor& p_tensor)
{
	for (int i = 0; i < p_tensor._shape[0]; i++) {
		for (int j = 0; j < p_tensor._shape[1]; j++) {
			for (int k = 0; k < p_tensor._shape[2]; k++) {
				if (k == p_tensor._shape[2] - 1) {
					output << p_tensor._data[i * p_tensor._shape[1] * p_tensor._shape[2] + j * p_tensor._shape[2] + k];
				}
				else {
					output << p_tensor._data[i * p_tensor._shape[1] * p_tensor._shape[2] + j * p_tensor._shape[2] + k] << ",";
				}
			}
			output << std::endl;
		}
		output << std::endl;
	}
}

void tensor::print_batch_volume(std::ostream& output, const tensor& p_tensor)
{
	for (int n = 0; n < p_tensor._shape[0]; n++) {
		for (int i = 0; i < p_tensor._shape[1]; i++) {
			for (int j = 0; j < p_tensor._shape[2]; j++) {
				for (int k = 0; k < p_tensor._shape[3]; k++) {
					if (k == p_tensor._shape[3] - 1) {
						output << p_tensor._data[n * p_tensor._shape[1] * p_tensor._shape[2] * p_tensor._shape[3] + i * p_tensor._shape[2] * p_tensor._shape[3] + j * p_tensor._shape[3] + k];
					}
					else {
						output << p_tensor._data[n * p_tensor._shape[1] * p_tensor._shape[2] * p_tensor._shape[3] + i * p_tensor._shape[2] * p_tensor._shape[3] + j * p_tensor._shape[3] + k] << ",";
					}
				}
				output << std::endl;
			}
			output << std::endl;
		}
	}
}

void tensor::check_gpu(const tensor& p_lhs, const tensor& p_rhs)
{
	if ((p_lhs._gpu_flag && !p_rhs._gpu_flag) || (!p_lhs._gpu_flag && p_rhs._gpu_flag))
	{
		assert(("Invalid binary operation on GPU: only one tensor is loaded to GPU memory", 0));
	}
}

int tensor::check_shape(std::initializer_list<int>& p_shape) const
{
	int result = SHAPE_EQUAL;
	if (_rank != p_shape.size())
	{
		result = SHAPE_DIFF;
	}
	else
	{
		auto it = p_shape.begin();
		for(int i = 0; i < p_shape.size(); i++)
		{
			if (_shape[i] != *it++)
			{
				result = SHAPE_EQUAL_DIFF_SIZE;
			}			
		}
	}

	return result;
}

int tensor::check_shape(const int p_rank, const int* p_shape) const
{
	int result = SHAPE_EQUAL;
	if (_rank != p_rank)
	{
		result = SHAPE_DIFF;
	}
	else
	{
		for (int i = 0; i < p_rank; i++)
		{
			if (_shape[i] != p_shape[i])
			{
				result = SHAPE_EQUAL_DIFF_SIZE;
			}
		}
	}

	return result;
}

int* tensor::init_shape(int &p_rank, std::initializer_list<int>& p_shape)
{
	int *result = nullptr;
	if (p_shape.size() == p_rank)
	{
		result = static_cast<int*>(malloc(sizeof(int) * p_rank));
		int index = 0;
		for (auto shape : p_shape)
		{
			result[index] = shape;
			index++;
		}
	}
	else
	{
		assert(0);
	}

	return result;

}

int* tensor::init_shape(int& p_rank, int* p_shape)
{
	int *result = static_cast<int*>(malloc(sizeof(int) * p_rank));
	memcpy(result, p_shape, sizeof(int) * p_rank);

	return result;
}

int tensor::init_size(int& p_rank, const int* p_shape)
{
	int result = 1;
	for(int i = 0; i < p_rank; i++)
	{
		result *= p_shape[i];
	}

	return result;
}

int* tensor::init_stride(int& p_rank, const int* p_shape)
{
	int *result = static_cast<int*>(malloc(sizeof(int) * p_rank));
	int stride = 1;

	for (int i = p_rank-1; i >= 0; i--)
	{
		result[i] = stride;
		stride *= p_shape[i];
	}

	return result;
}

float* tensor::init_data(int& p_size, INIT p_init, float p_value)
{
	float* result = static_cast<float*>(malloc(sizeof(float) * p_size));

	fill(result, p_size, p_init, p_value);

	return result;
}

void tensor::fill(float* p_data, const int p_size, const INIT p_init, const float p_value)
{
	float* x = p_data;
	switch (p_init)
	{
	case NONE:
		{			
		}
		break;
	case ZERO:
		{
			for (int i = 0; i < p_size; i++)
			{
				*x++ = 0.f;
			}
		}
		break;
	case VALUE:
		{
			for (int i = 0; i < p_size; i++)
			{
				*x++ = p_value;
			}
		}
		break;
	default:;
	}
}

float* tensor::gpu_data() const
{
	return _gpu_data->_data;
}

tensor operator+(const float p_lhs, const tensor& p_rhs)
{
	return p_rhs + p_lhs;
}

tensor operator-(const float p_lhs, const tensor& p_rhs)
{
	tensor result(p_rhs._rank, p_rhs._shape);

	if (p_rhs._gpu_flag)
	{
		result.to_gpu();
		tensor_operator_gpu::const_sub(p_lhs, p_rhs.gpu_data(), result.gpu_data(), p_rhs._size);
	}
	else
	{
		tensor_operator_cpu::const_sub(p_lhs, p_rhs._data, result._data, p_rhs._size);
	}
	return result;
}

tensor operator*(const float p_lhs, const tensor& p_rhs)
{
	return p_rhs * p_lhs;
}

tensor operator/(const float p_lhs, const tensor& p_rhs)
{
	tensor result(p_rhs._rank, p_rhs._shape);

	if (p_rhs._gpu_flag)
	{
		result.to_gpu();
		tensor_operator_gpu::const_div(p_lhs, p_rhs.gpu_data(), result.gpu_data(), p_rhs._size);
	}
	else
	{
		tensor_operator_cpu::const_div(p_lhs, p_rhs._data, result._data, p_rhs._size);
	}
	return result;
}