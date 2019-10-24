#include "Tensor.h"
#include "RandomGenerator.h"
#include <cassert>
#include <iostream>
#include "TensorPool.h"
#include "TensorOperator.h"

bool Tensor::pooling = true;

Tensor::Tensor(): _arr(nullptr), _rank(0), _shape(nullptr), _size(0), _end(0), _transpose(false)
{
}

Tensor::Tensor(const initializer_list<int> p_shape, const INIT p_init, const float p_value) {
	_transpose = false;
	init_shape(p_shape);
	_arr = alloc_arr(_size);
	_end = 0;
	fill(p_init, p_value);
}

Tensor::Tensor(const initializer_list<int> p_shape, float* p_data) {
	_transpose = false;
	init_shape(p_shape);
	_arr = p_data;
	_end = 0;
}

Tensor::Tensor(const initializer_list<int> p_shape, const vector<uint8_t>& p_data)
{
	_transpose = false;
	init_shape(p_shape);
	_arr = alloc_arr(_size);
	_end = 0;

	for (int i = 0; i < p_data.size(); i++)
	{
		_arr[i] = p_data.at(i);
	}
}

Tensor::Tensor(const int p_rank, int* p_shape, float* p_data) {
	_transpose = false;
	init_shape(p_rank, p_shape, false);
	_arr = p_data;
	_end = 0;
}

Tensor::Tensor(const int p_rank, int* p_shape, const INIT p_init, const float p_value) {
	_transpose = false;
	init_shape(p_rank, p_shape, true);
	_arr = alloc_arr(_size);
	_end = 0;
	fill(p_init, p_value);
}

Tensor::Tensor(const initializer_list<int> p_shape, initializer_list<float> p_inputs) {
	_transpose = false;
	init_shape(p_shape);
	_arr = alloc_arr(_size);
	_end = 0;

	int i = 0;
	for (auto p_input : p_inputs)
	{
		_arr[i] = p_input;
		i++;
	}
}

Tensor::Tensor(const Tensor& p_copy) {
	_transpose = p_copy._transpose;;
	_rank = p_copy._rank;
	_shape = alloc_shape(_rank);
	_size = p_copy._size;
	_arr = alloc_arr(_size);
	_end = p_copy._end;

	memcpy(_shape, p_copy._shape, sizeof(int) * _rank);
	memcpy(_arr, p_copy._arr, sizeof(float) * static_cast<size_t>(_size));
}

Tensor::~Tensor() {
	if (_shape != nullptr) free_shape();
	_shape = nullptr;
	if (_arr != nullptr) free_arr();
	_arr = nullptr;
	_rank = -1;
}

Tensor Tensor::operator-() const {
	float* arr = alloc_arr(_size);
	int* shape = copy_shape(_rank, _shape);

	for (int i = 0; i < _size; i++) {
		arr[i] = -_arr[i];
	}

	return Tensor(_rank, shape, arr);
}

Tensor Tensor::Zero(const initializer_list<int> p_shape) {
	return Tensor(p_shape, ZERO);
}

Tensor Tensor::Ones(const initializer_list<int> p_shape) {
	return Tensor(p_shape, ONES);
}

Tensor Tensor::Value(const initializer_list<int> p_shape, const float p_value) {
	return Tensor(p_shape, VALUE, p_value);
}

Tensor Tensor::Random(const initializer_list<int> p_shape, const float p_limit) {
	return Tensor(p_shape, RANDOM, p_limit);
}

Tensor& Tensor::operator=(const Tensor& p_copy) {
	_transpose = p_copy._transpose;
	if (_rank != p_copy._rank) {
		if (_shape != nullptr) {
			free_shape();
		}

		_rank = p_copy._rank;
		_shape = alloc_shape(_rank);
	}

	if (_size != p_copy._size) {
		if (_arr != nullptr) {
			free_arr();
		}

		_size = p_copy._size;
		_arr = alloc_arr(_size);
	}

	_end = p_copy._end;

	memcpy(_shape, p_copy._shape, sizeof(int) * _rank);
	memcpy(_arr, p_copy._arr, sizeof(float) * _size);

	return *this;
}

float& Tensor::operator[](const int p_index) const {
	return _arr[p_index];
}

Tensor Tensor::T()
{
	Tensor result = *this;
	result._transpose = !_transpose;
	return result;
}

Tensor Tensor::vec() const
{
	Tensor result = *this;
	result._rank = 1;
	const int shape = result._shape[0] * result._shape[1];
	result.free_shape();
	result.init_shape({shape});
	return result;
}

Tensor& Tensor::operator+=(const Tensor& p_rhs)
{
	check_size_eq(p_rhs._size);

	TensorOperator::instance().vv_add(_arr, p_rhs._arr, _arr, _size);
	return *this;
}

Tensor& Tensor::operator+=(const float p_rhs)
{
	float* x = &_arr[0];

	for (int i = 0; i < _size; i++)
	{
		*x++ += p_rhs;
	}
	
	return *this;
}

Tensor& Tensor::operator-=(const Tensor& p_rhs)
{
	check_size_eq(p_rhs._size);

	TensorOperator::instance().vv_sub(_arr, p_rhs._arr, _arr, _size);
	return *this;
}

Tensor& Tensor::operator-=(const float p_rhs)
{
	*this += -p_rhs;
	return *this;
}

Tensor Tensor::operator*=(const Tensor& p_rhs) const
{	
	Tensor result;
	int rows;
	int cols;
	int common_l;
	int common_r;
	
	if (_rank == 1 && p_rhs._rank == 1)
	{
		rows = _transpose ? 1 : _shape[0];
		common_l = _transpose ? _shape[0] : 1;
		cols = p_rhs._transpose ? p_rhs._shape[0] : 1;
		common_r = p_rhs._transpose ? 1 : p_rhs._shape[0];
#ifdef _DEBUG
		if (common_l != common_r)
		{
			assert(("Invalid tensor product (common_l != common_r)", 0));
		}
#endif		
		result = Zero({ rows, cols });
		TensorOperator::instance().MM_prod(_arr, _transpose, p_rhs._arr, p_rhs._transpose, result._arr, rows, common_l, cols);
	}
	if (_rank == 1 && p_rhs._rank == 2)
	{
		rows = _transpose ? 1 : _shape[0];
		common_l = _transpose ? _shape[0] : 1;
		cols = p_rhs._transpose ? p_rhs._shape[0] : p_rhs._shape[1];
		common_r = p_rhs._transpose ? p_rhs._shape[1] : p_rhs._shape[0];
#ifdef _DEBUG
		if (common_l != common_r)
		{
			assert(("Invalid tensor product (common_l != common_r)", 0));
		}
#endif
		result = Zero({ rows, cols });
		TensorOperator::instance().MM_prod(_arr, _transpose, p_rhs._arr, p_rhs._transpose, result._arr, rows, common_l, cols);
	}
	if (_rank == 2 && p_rhs._rank == 1)
	{
		rows = _transpose ? _shape[1] : _shape[0];
		common_l = _transpose ? _shape[0] : _shape[1];
		cols = p_rhs._transpose ? p_rhs._shape[0] : 1;
		common_r = p_rhs._transpose ? 1 : p_rhs._shape[0];
#ifdef _DEBUG
		if (common_l != common_r)
		{
			assert(("Invalid tensor product (common_l != common_r)", 0));
		}		
#endif
		result = Zero({ rows, cols });
		TensorOperator::instance().MM_prod(_arr, _transpose, p_rhs._arr, p_rhs._transpose, result._arr, rows, common_l, cols);
	}
	if (_rank == 2 && p_rhs._rank == 2)
	{
		rows = _transpose ? _shape[1] : _shape[0];
		common_l = _transpose ? _shape[0] : _shape[1];
		cols = p_rhs._transpose ? p_rhs._shape[0] : p_rhs._shape[1];
		common_r = p_rhs._transpose ? p_rhs._shape[1] : p_rhs._shape[0];
#ifdef _DEBUG
		if (common_l != common_r)
		{
			assert(("Invalid tensor product (common_l != common_r)", 0));
		}
#endif
		result = Zero({rows, cols});
		TensorOperator::instance().MM_prod(_arr, _transpose, p_rhs._arr, p_rhs._transpose, result._arr, rows, common_l, cols);
	}
	if (_rank > 2 || p_rhs._rank > 2)
	{
#ifdef _DEBUG
		assert(("Invalid tensor product (_rank > 2)", 0));
#endif		
	}

	return result;
}

Tensor& Tensor::operator*=(const float p_rhs)
{
	float* x = &_arr[0];

	for (int i = 0; i < _size; i++)
	{
		*x++ *= p_rhs;
	}

	return *this;
}

Tensor& Tensor::operator/=(const float p_rhs)
{
	*this *= 1.f / p_rhs;
	return *this;
}

Tensor Tensor::operator+(const Tensor& p_rhs) const
{
	check_size_eq(_size);
	Tensor result = *this;
	return result += p_rhs;
}

Tensor Tensor::operator+(const float p_rhs) const
{
	Tensor result = *this;
	return result += p_rhs;
}

Tensor Tensor::operator-(const Tensor& p_rhs) const
{
	check_size_eq(_size);
	Tensor result = *this;
	return result -= p_rhs;
}

Tensor Tensor::operator-(const float p_rhs) const
{
	Tensor result = *this;
	return result -= p_rhs;
}

Tensor Tensor::operator*(const Tensor& p_rhs) const
{
	Tensor result = *this;
	return result *= p_rhs;
}

Tensor Tensor::operator*(const float p_rhs) const
{
	Tensor result = *this;
	return result *= p_rhs;
}


void Tensor::get_row(Tensor& p_tensor, const int p_row) const
{
	if (p_tensor.size() != _shape[1])
	{
		assert(0);
	}

	for(int i = 0; i < _shape[1]; i++)
	{
		p_tensor[i] = _arr[p_row * _shape[1] + i];
	}
}

void Tensor::set_row(Tensor& p_tensor, const int p_row) const
{
	if (p_tensor.size() != _shape[1])
	{
		assert(0);
	}

	for (int i = 0; i < _shape[1]; i++)
	{
		_arr[p_row * _shape[1] + i] = p_tensor[i];
	}
}

void Tensor::get_column(Tensor& p_tensor, const int p_column) const
{
	if (p_tensor.size() != _shape[0])
	{
		assert(0);
	}

	for (int i = 0; i < _shape[0]; i++)
	{
		p_tensor[i] = _arr[i * _shape[1] + p_column];
	}
}

void Tensor::set_column(Tensor& p_tensor, const int p_column) const
{
	if (p_tensor.size() != _shape[0])
	{
		assert(0);
	}

	for (int i = 0; i < _shape[0]; i++)
	{
		_arr[i * _shape[1] + p_column] = p_tensor[i];
	}
}

int Tensor::max_value_index() const {
	int max = 0;

	for (int i = 0; i < _size; i++) {
		if (_arr[max] < _arr[i]) max = i;
	}

	return max;
}

void Tensor::override(Tensor* p_tensor) const {
	memcpy(_arr, p_tensor->_arr, sizeof(float) * static_cast<size_t>(_size));
}

void Tensor::override(const float* p_data) const
{
	for (int i = 0; i < _size; i++) {
		_arr[i] = p_data[i];
	}
}

void Tensor::override(const int* p_data) const
{
	for (int i = 0; i < _size; i++) {
		_arr[i] = static_cast<float>(p_data[i]);
	}
}

void Tensor::fill(const float p_value) const {
	if (p_value == 0)
	{
		fill(ZERO, 0);
	}
	else
	{
		fill(VALUE, p_value);
	}	
}

float Tensor::at(const int p_x) const {
	check_size_gt(p_x);
	return _arr[p_x];
}

float Tensor::at(const int p_y, const int p_x) const {
	const int index = p_y * _shape[1] + p_x;

	check_size_gt(index);

	return _arr[index];
}

float Tensor::at(const int p_z, const int p_y, const int p_x) const {
	return _arr[p_z * _shape[2] * _shape[1] + p_y * _shape[1] + p_x];
}

void Tensor::set(const int p_x, const float p_val) const {
	check_size_gt(p_x);

	_arr[p_x] = p_val;
}

void Tensor::set(const int p_y, const int p_x, const float p_val) const {
	const int index = p_y * _shape[1] + p_x;

	check_size_gt(index);

	_arr[index] = p_val;
}

void Tensor::set(const int p_z, const int p_y, const int p_x, const float p_val) const {
	_arr[p_z * _shape[2] * _shape[1] + p_y * _shape[1] + p_x] = p_val;
}

float Tensor::element_prod() const
{
	float result = 1;

	for(int i = 0; i < _size; i++)
	{
		result *= _arr[i];
	}

	return result;
}

float* Tensor::alloc_arr(const int p_size) {
	float* result = nullptr;

	if (pooling)
	{
		result = TensorPool::instance().get(p_size);
	}
	else
	{
		result = static_cast<float*>(malloc(p_size * sizeof(float)));
	}

	return result;
}

int* Tensor::alloc_shape(const int p_size) {
	return static_cast<int*>(malloc(p_size * sizeof(int)));
}

void Tensor::free_arr() const {
	if (pooling)
	{
		TensorPool::instance().release(_size, _arr);
	}
	else
	{
		free(_arr);
	}
}

void Tensor::free_shape() const {
	free(_shape);
}

void Tensor::print_vector(ostream &output, const Tensor& p_tensor, bool p_cm)
{
	for (int i = 0; i < p_tensor._size; i++) {
		if (i == p_tensor._size - 1) {
			if (p_cm)
			{
				output << p_tensor._arr[i];
			}
			else
			{
				output << p_tensor._arr[i];
			}

		}
		else {
			if (p_cm)
			{
				output << p_tensor._arr[i] << ",";
			}
			else
			{
				output << p_tensor._arr[i] << ",";
			}
		}
	}
}

void Tensor::print_matrix(ostream &output, const Tensor& p_tensor, bool p_cm)
{
	for (int i = 0; i < p_tensor._shape[0]; i++) {
		for (int j = 0; j < p_tensor._shape[1]; j++)
		{
			if (j == p_tensor._shape[1] - 1) {
				if (p_cm)
				{
					output << p_tensor._arr[j * p_tensor._shape[0] + i];
				}
				else
				{
					output << p_tensor._arr[i * p_tensor._shape[1] + j];
				}
			}
			else {
				if (p_cm)
				{
					output << p_tensor._arr[j * p_tensor._shape[0] + i] << ",";
				}
				else
				{
					output << p_tensor._arr[i * p_tensor._shape[1] + j] << ",";
				}
			}
		}
		output << endl;
	}
}

void Tensor::print_volume(ostream& output, const Tensor& p_tensor)
{
	for (int i = 0; i < p_tensor._shape[0]; i++) {
		for (int j = 0; j < p_tensor._shape[1]; j++) {
			for (int k = 0; k < p_tensor._shape[2]; k++) {
				if (k == p_tensor._shape[2] - 1) {
					output << p_tensor._arr[i * p_tensor._shape[1] * p_tensor._shape[2] +  j * p_tensor._shape[2] + k];
				}
				else {
					output << p_tensor._arr[i * p_tensor._shape[1] * p_tensor._shape[2] + j * p_tensor._shape[2] + k] << ",";
				}
			}
			output << endl;
		}
		output << endl;
	}
}

int* Tensor::copy_shape(const int p_rank, const int* p_shape) {
	int* shape = alloc_shape(p_rank);

	for (int i = 0; i < p_rank; i++) {
		shape[i] = p_shape[i];
	}

	return shape;
}

void Tensor::push_back(Tensor* p_tensor)
{
	check_size_gt(_end + p_tensor->_size);

	if (_rank == 1)
	{
		memcpy((_arr + _end), p_tensor->_arr, sizeof(float) * p_tensor->_size);
		_end += p_tensor->_size;
	}
	if (_rank == 2)
	{
		int rows = 0;
		int cols = 0;

		if (p_tensor->_rank == 1)
		{
			rows = 1;
			cols = p_tensor->_shape[0];
		}
		if (p_tensor->_rank == 2)
		{
			rows = p_tensor->_shape[0];
			cols = p_tensor->_shape[1];
		}

		for(int i = 0; i < rows; i++)
		{
			memcpy((_arr + (i * _shape[1] + _end)), (p_tensor->_arr + i * cols), sizeof(float) * cols);
		}
		_end += cols;
	}
	if (_rank == 3)
	{
		int depth = 0;
		int rows = 0;
		int cols = 0;

		if (p_tensor->_rank == 1)
		{
			depth = 1;
			rows = 1;
			cols = p_tensor->_shape[0];
		}
		if (p_tensor->_rank == 2)
		{
			depth = 1;
			rows = p_tensor->_shape[0];
			cols = p_tensor->_shape[1];
		}
		if (p_tensor->_rank == 3)
		{
			depth = p_tensor->_shape[0];
			rows = p_tensor->_shape[1];
			cols = p_tensor->_shape[2];
		}

		for (int i = 0; i < depth; i++)
		{
			const int index = (i * _shape[1] * _shape[2] + _end);
			memcpy((_arr + index), (p_tensor->_arr + i * rows * cols), sizeof(float) * rows * cols);
		}
		_end += rows * cols;
	}
}

void Tensor::splice(const int p_start, Tensor* p_output) const
{
	if (_rank == 1)
	{
		memcpy(p_output->_arr, _arr + p_start, sizeof(float) * p_output->_size);
	}
	if (_rank == 2)
	{
		int rows = 0;
		int cols = 0;

		if (p_output->_rank == 1)
		{
			rows = 1;
			cols = p_output->_shape[0];
		}
		if (p_output->_rank == 2)
		{
			rows = p_output->_shape[0];
			cols = p_output->_shape[1];
		}

		for (int i = 0; i < rows; i++)
		{
			memcpy(p_output->_arr + i * cols, _arr + (i * _shape[1] + p_start), sizeof(float) * cols);
		}
	}
	
}

void Tensor::reset_index()
{
	_end = 0;
}

Tensor* Tensor::concat(vector<Tensor*>& p_input)
{
	int dim = 0;
	for(auto t : p_input)
	{
		dim += t->size();
	}

	Tensor* result = new Tensor({ dim }, ZERO);

	for(auto t : p_input)
	{
		result->push_back(t);
	}
	result->reset_index();

	return result;
}

Tensor Tensor::concat(vector<Tensor>& p_vector)
{
	int size = 0;
	for (auto& v : p_vector)
	{
		size += v.size();
	}

	float *res = alloc_arr(size);

	int i = 0;

	for (auto& v : p_vector)
	{
		for (int j = 0; j < v.size(); j++)
		{
			res[i] = v[j];
			i++;
		}
	}

	return Tensor({ size }, res);
}

void Tensor::padding(const int p_padding)
{
	float* arr = nullptr;

	if (_rank == 1)
	{
		arr = alloc_arr(_shape[0] + 2 * p_padding);
		memset(arr, 0, sizeof(float) * (_shape[0] + 2 * p_padding));

		if (p_padding > 0)
		{
			memcpy(arr + p_padding, _arr, sizeof(float) * _size);
		}
		if (p_padding < 0)
		{
			memcpy(arr, _arr - p_padding, sizeof(float) * (_size - 2 * p_padding)); 
		}		

		_shape[0] += 2 * p_padding;
		_size = _shape[0];
	}

	if (_rank == 2)
	{
		arr = alloc_arr((_shape[0] + 2 * p_padding) * (_shape[1] + 2 * p_padding));
		memset(arr, 0, sizeof(float) * (_shape[0] + 2 * p_padding) * (_shape[1] + 2 * p_padding));

		for(int i = 0; i < _shape[0]; i++)
		{
			const int index = (i + p_padding) * (_shape[1] + 2 * p_padding) + p_padding;
			memcpy(arr + index, _arr + i * _shape[1], sizeof(float) * _shape[1]);
		}

		_shape[0] += 2 * p_padding;
		_shape[1] += 2 * p_padding;
		_size = _shape[0] * _shape[1];
	}

	delete _arr;
	_arr = arr;

}

void Tensor::reshape(const initializer_list<int> p_shape)
{
	free_shape();
	init_shape(p_shape);
}

Tensor Tensor::slice(const int p_index) const
{
	const int rank = 2;
	float* arr = alloc_arr(_shape[1] * _shape[2]);
	int* shape = alloc_shape(rank);

	if (_rank == 3)
	{
		shape[0] = _shape[1];
		shape[1] = _shape[2];

		const int index = p_index * _shape[1] * _shape[2];
		memcpy(arr, _arr + index, sizeof(float) * _shape[1] * _shape[2]);
	}

	return Tensor(rank, shape, arr);
}

void Tensor::replicate(const int p_n)
{
	float* arr = alloc_arr(_size * p_n);
	float* ax = &arr[0];
	float* sx = &_arr[0];

	for(int i = 0; i < _shape[0]; i++)
	{
		for(int j = 0; j < _shape[1]; j++)
		{
			for(int k = 0; k < p_n; k++)
			{
				*ax++ = *sx;
			}
			sx++;
		}
	}

	free_arr();
	_arr = arr;
	_size *= p_n;
	_shape[1] *= p_n;

}

void Tensor::subregion(Tensor* p_dest, Tensor* p_source, const int p_y, const int p_x, const int p_h, const int p_w)
{
#ifdef _DEBUG
	if (p_dest->_size != p_h * p_w)
	{
		assert(("Invalid size", 0));
	}
#endif

	for(int i = 0; i < p_h; i++)
	{
		const int index = (p_y + i) * p_source->_shape[1] + p_x;
		memcpy(p_dest->_arr + (i * p_w), p_source->_arr + index, sizeof(float) * p_w);
	}
}

void Tensor::subregion(Tensor* p_dest, Tensor* p_source, const int p_z, const int p_y, const int p_x, const int p_h, const int p_w)
{
#ifdef _DEBUG
	if (p_dest->_size != p_h * p_w)
	{
		assert(("Invalid size", 0));
	}
	if (p_source->_shape[0] < p_z)
	{
		assert(("Invalid depth", 0));
	}
	if (p_source->_shape[1] < p_y + p_h)
	{
		assert(("Invalid height", 0));
	}
	if (p_source->_shape[2] < p_x + p_w)
	{
		assert(("Invalid width", 0));
	}
#endif

	for (int i = 0; i < p_h; i++)
	{
		const int index = (p_z * p_source->_shape[1] * p_source->_shape[2]) + (p_y + i) * p_source->_shape[2] + p_x;
		memcpy(p_dest->_arr + (i * p_w), p_source->_arr + index, sizeof(float) * p_w);
	}
}

void Tensor::add_subregion(Tensor* p_dest, int p_yd, int p_xd, Tensor* p_source, int p_y, int p_x, int p_h, int p_w)
{
#ifdef _DEBUG
	if (p_dest->_shape[0] < p_yd + p_h)
	{
		assert(("Insufficient rows", 0));
	}
	if (p_dest->_shape[1] < p_xd + p_w)
	{
		assert(("Insufficient cols", 0));
	}
#endif

	for(int i = 0; i < p_h; i++)
	{
		for(int j = 0; j < p_w; j++)
		{
			p_dest->_arr[(p_yd + i) * p_dest->_shape[1] + p_xd + j] += p_source->_arr[(p_y + i) * p_source->_shape[1] + p_x + j];
		}
	}
}

void Tensor::add_subregion(Tensor* p_dest, int p_zd, int p_yd, int p_xd, int p_hd, int p_wd, Tensor* p_source, int p_y, int p_x, int p_h, int p_w)
{
#ifdef _DEBUG
	if (p_hd * p_wd != p_h * p_w)
	{
		assert(("Invalid shape dimension", 0));
	}
	if (p_source->_shape[0] < p_y + p_h)
	{
		assert(("Invalid source position (rows)", 0));
	}
	if (p_source->_shape[1] < p_x + p_w)
	{
		assert(("Invalid source position (cols)", 0));
	}
	if (p_dest->_shape[0] < p_zd)
	{
		assert(("Invalid destination position (depth)", 0));
	}
	if (p_dest->_shape[1] < p_yd + p_hd)
	{
		assert(("Invalid destination position (rows)", 0));
	}
	if (p_dest->_shape[2] < p_xd + p_wd)
	{
		assert(("Invalid destination position (cols)", 0));
	}
#endif

	int id = 0;
	int jd = 0;

	float* x = &p_source->_arr[p_y * p_source->_shape[1] + p_x];
	float* y = &p_dest->_arr[p_zd * p_dest->_shape[1] * p_dest->_shape[2] + p_yd * p_dest->_shape[2] + p_xd];

	for (int i = 0; i < p_h; i++)
	{
		for (int j = 0; j < p_w; j++)
		{
			//p_dest->_arr[p_zd * p_dest->_shape[1] * p_dest->_shape[2] + (p_yd + id) * p_dest->_shape[2] + p_xd + jd] += p_source->_arr[(p_y + i) * p_source->_shape[1] + p_x + j];
			*y++ += *x++;
			jd++;

			if (jd == p_wd)
			{
				jd = 0;
				id++;
				y += p_dest->_shape[2] - p_wd;
			}
		}
		x += p_source->_shape[1] - p_w;
	}
}

int Tensor::subregion_max_index(Tensor* p_source, const int p_y, const int p_x, const int p_h, const int p_w)
{
	int index = p_y * p_source->_shape[1] + p_x;
	int result = index;

	for (int i = 0; i < p_h; i++)
	{
		for (int j = 0; j < p_w; j++)
		{
			index = (p_y + i) * p_source->_shape[1] + p_x + j;

			if (p_source->_arr[result] < p_source->_arr[index])
			{
				result = index;
			}
		}
	}

	return result;
}

void Tensor::slice(Tensor* p_dest, Tensor* p_source, const int p_index)
{
	if (p_source->rank() == 3)
	{
		const int index = p_index * p_source->shape(1) * p_source->shape(2);
		memcpy(p_dest->_arr, p_source->_arr + index, sizeof(float) * p_source->shape(1) * p_source->shape(2));
	}
}

void Tensor::init_shape(const int p_rank, int* p_shape, const bool p_copy_shape) {
	_rank = p_rank;
	_shape = p_copy_shape ? copy_shape(p_rank, p_shape) : p_shape;
	_size = 1;

	for(int i = 0; i < _rank; i++) {
		_size *= _shape[i];
	}
}

void Tensor::init_shape(initializer_list<int> p_shape) {
	_rank = p_shape.size();
	_shape = alloc_shape(_rank);
	_size = 1;

	int i = 0;
	for (auto s = p_shape.begin(); s != p_shape.end(); s++) {
		_shape[i] = *s;
		_size *= _shape[i];
		i++;
	}
}

void Tensor::fill(const INIT p_init, const float p_value) const {
	switch (p_init) {
		case ZERO:
			memset(_arr, 0, sizeof(float) * _size);
			break;
		case ONES:
			if (_rank == 1) {
				fill(VALUE, 1);
			}
			if (_rank == 2) {				
				if (_shape[0] == _shape[1]) {
					fill(0);
					for (int i = 0; i < _shape[0]; i++) _arr[i * _shape[1] + i] = 1;
				}
				else
				{
					memset(_arr, 1, sizeof(float) * _size);
				}
			}			
			break;
		case VALUE:
			for (int i = 0; i < _size; i++) _arr[i] = p_value;
			break;
		case RANDOM:
			for (int i = 0; i < _size; i++) _arr[i] = RandomGenerator::get_instance().random(-p_value, p_value);
			break;
	}
}

void Tensor::check_size_gt(const int p_size) const
{
#ifdef _DEBUG
	if (p_size > _size)
	{
		assert(("Size limit", 0));
	}
#endif
}

void Tensor::check_size_eq(const int p_size) const
{
#ifdef _DEBUG
	if (p_size != _size)
	{
		assert(("Invalid size", 0));
	}
#endif
}

void Tensor::check_rank_gt(const int p_rank) const
{
#ifdef _DEBUG
	if (p_rank > _rank)
	{
		assert(("Invalid rank", 0));
	}
#endif
}

void Tensor::check_rank_eq(const int p_rank) const
{
#ifdef _DEBUG
	if (p_rank != _rank)
	{
		assert(("Invalid rank", 0));
	}
#endif
}