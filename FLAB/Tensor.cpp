#include "Tensor.h"
#include "RandomGenerator.h"
#include <ppl.h>
#include <cassert>

using namespace FLAB;
using namespace Concurrency;

Tensor::Tensor(): _arr(nullptr), _rank(0), _shape(nullptr), _size(0) {
}

Tensor::Tensor(const initializer_list<int> p_shape, const INIT p_init, const float p_value) {
	init_shape(p_shape);
	_arr = alloc_arr(_size);
	fill(p_init, p_value);
}

Tensor::Tensor(const initializer_list<int> p_shape, float* p_data) {
	init_shape(p_shape);
	_arr = p_data;
}

Tensor::Tensor(const int p_rank, int* p_shape, float* p_data) {
	init_shape(p_rank, p_shape, false);
	_arr = p_data;
}

Tensor::Tensor(const int p_rank, int* p_shape, const INIT p_init, const float p_value) {
	init_shape(p_rank, p_shape, true);
	_arr = alloc_arr(_size);
	fill(p_init, p_value);
}

Tensor::Tensor(const initializer_list<int> p_shape, initializer_list<float> p_inputs) {
	init_shape(p_shape);
	_arr = alloc_arr(_size);

	int i = 0;
	for(auto v = p_inputs.begin(); v != p_inputs.end(); v++) {
		_arr[i] = *v;
		i++;
	}
}

Tensor::Tensor(const Tensor& p_copy) {
	_rank = p_copy._rank;
	_shape = alloc_shape(_rank);
	_size = p_copy._size;
	_arr = alloc_arr(_size);

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

void Tensor::operator=(const Tensor& p_copy) {
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

	memcpy(_shape, p_copy._shape, sizeof(int) * _rank);
	memcpy(_arr, p_copy._arr, sizeof(float) * _size);
}

Tensor Tensor::operator+(const Tensor& p_tensor) const {
	Tensor temp(*this);

	return temp += p_tensor;
}

Tensor Tensor::operator+(const float p_const) const {
	Tensor temp(*this);

	return temp += p_const;
}

Tensor& Tensor::operator+=(const Tensor& p_tensor) {
	if (_size != p_tensor.size() || _rank != p_tensor._rank)
	{
		assert(("Size or rank not equal", 0));
	}

	float *xpos = &p_tensor._arr[0];
	float *ypos = &_arr[0];

	for (int i = 0; i < _size; i++) {
		(*ypos++) += (*xpos++);
	}

	return *this;
}

Tensor& Tensor::operator+=(const float p_const) {
	for (int i = 0; i < _size; i++) {
		_arr[i] += p_const;
	}

	return *this;
}

Tensor Tensor::operator-(const Tensor& p_tensor) const {
	Tensor temp(*this);

	return temp -= p_tensor;
}

Tensor& Tensor::operator-=(const Tensor& p_tensor) {
	if (_size != p_tensor.size() || _rank != p_tensor._rank)
	{
		assert(("Size or rank not equal", 0));
	}

	float *xpos = &p_tensor._arr[0];
	float *ypos = &_arr[0];

	for (int i = 0; i < _size; i++) {
		(*ypos++) -= (*xpos++);
	}

	return *this;
}

Tensor Tensor::operator*(const Tensor& p_tensor) const {
	float* arr = nullptr;
	int rank = 0;
	int* shape = nullptr;
	
	if (this->_rank == 1 && p_tensor._rank == 1) {
		if (_size != p_tensor.size())
		{
			assert(("Size not equal", 0));
		}

		arr = dot(this, &p_tensor);
		rank = 1;
		shape = copy_shape(_rank, _shape);
	}

	if (this->_rank == 2 && p_tensor._rank == 1 && this->shape(1) == p_tensor.shape(0)) {
		arr = mat_vec(this, &p_tensor);
		rank = 1;
		shape = alloc_shape(rank);
		shape[0] = _shape[0];
	}

	if (this->_rank == 1 && p_tensor._rank == 2)
	{
		assert(0);
	}

	if (this->_rank == 2 && p_tensor._rank == 2) { // preverit spravnu funkcnost
		arr = alloc_arr(_shape[0] * p_tensor._shape[1]);
		rank = 2;
		shape = alloc_shape(rank);
		shape[0] = _shape[0];
		shape[1] = p_tensor._shape[1];

		int rows = _shape[0];
		int rows2 = p_tensor._shape[0];
		int cols = _shape[1];
		int cols2 = p_tensor._shape[1];

		for (int i = 0; i < shape[0]; i++) {
			for (int j = 0; j < shape[1]; j++) {
				arr[i * shape[1] + j] = 0;
				for (int k = 0; k < cols; k++) {
					arr[i * shape[1] + j] += _arr[i * cols + k] * p_tensor._arr[k * cols2 + j];
				}
			}
		}
	}

	return Tensor(rank, shape, arr);
}

Tensor Tensor::operator*(const float p_const) const {
	Tensor temp(*this);

	return temp *= p_const;
}

Tensor& Tensor::operator*=(const float p_const) {
	for (int i = 0; i < _size; i++) {
		_arr[i] *= p_const;
	}

	return *this;
}

Tensor Tensor::operator/(const float p_const) const {
	Tensor temp(*this);

	return temp /= p_const;
}

Tensor Tensor::operator/(const Tensor& p_tensor) const {
	float* arr = alloc_arr(_size);
	int* shape = copy_shape(_rank, _shape);

	for (int i = 0; i < _size; i++) {
		arr[i] = _arr[i] / p_tensor._arr[i];
	}

	return Tensor(_rank, shape, arr);
}

Tensor& Tensor::operator/=(const float p_const) {
	for (int i = 0; i < _size; i++) {
		_arr[i] /= p_const;
	}

	return *this;
}

float& Tensor::operator[](const int p_index) const {
	return _arr[p_index];
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

Tensor Tensor::T() const {
	float* arr = alloc_arr(_size);

	int* shape = copy_shape(_rank, _shape);

	if (_rank == 1) {
		for (int i = 0; i < _shape[0]; i++) {
			arr[i] = _arr[i];
		}
	}

	if (_rank == 2) {
		shape[0] = _shape[1];
		shape[1] = _shape[0];

		for (int i = 0; i < _shape[0]; i++) {
			for (int j = 0; j < _shape[1]; j++) {
				arr[j * _shape[0] + i] = _arr[i * _shape[1] + j];
			}
		}
	}

	return Tensor(_rank, shape, arr);
}

Tensor Tensor::diag() const {
	const int rank = 2;
	int* shape = alloc_shape(rank);
	float* arr = alloc_arr(_size *_size);

	shape[0] = _shape[0];
	shape[1] = _shape[0];

	for(int i = 0; i < _size; i++) {
		for (int j = 0; j < _size; j++) {
			arr[i * _size + j] = i == j ? _arr[i] : 0;
		}
	}

	return Tensor(rank, shape, arr);
}

Tensor Tensor::pow(const float p_y) const {
	float* arr = alloc_arr(_size);
	int* shape = copy_shape(_rank, _shape);

	for (int i = 0; i < _size; i++) {
		arr[i] = std::pow(_arr[i], p_y);
	}

	return Tensor(_rank, shape, arr);
}

Tensor Tensor::sqrt() const {
	float* arr = alloc_arr(_size);
	int* shape = copy_shape(_rank, _shape);

	for (int i = 0; i < _size; i++) {
		arr[i] = std::sqrt(_arr[i]);
	}

	return Tensor(_rank, shape, arr);
}

Tensor Tensor::dot(const Tensor& p_tensor) const {
	if (_size != p_tensor.size() || _rank != p_tensor._rank)
	{
		assert(("Size or rank not equal", 0));
	}

	float* arr = dot(this, &p_tensor);
	int* shape = copy_shape(_rank, _shape);

	return Tensor(_rank, shape, arr);
}

Tensor Tensor::outer_prod(const Tensor& p_tensor) const
{
	float* arr = nullptr;
	int rank = 0;
	int* shape = nullptr;

	if (this->_rank != 1 || p_tensor._rank != 1)
	{
		assert(("Rank not equal", 0));
	}

	const int rows = _size;
	const int cols = p_tensor._size;

	arr = alloc_arr(rows * cols);
	rank = 2;
	shape = alloc_shape(rank);
	shape[0] = rows;
	shape[1] = cols;

	int r = rows / 4;

	if (r > 0)
	{
		float *xpos0 = &_arr[0];
		float *xpos1 = &_arr[1];
		float *xpos2 = &_arr[2];
		float *xpos3 = &_arr[3];

		float *zpos0 = &arr[0];
		float *zpos1 = &arr[cols];
		float *zpos2 = &arr[cols * 2];
		float *zpos3 = &arr[cols * 3];

		for (int i = 0; i < r; i++) {
			float *ypos = &p_tensor._arr[0];
			for (int j = 0; j < cols; j++) {
				(*zpos0++) = (*xpos0) * (*ypos);
				(*zpos1++) = (*xpos1) * (*ypos);
				(*zpos2++) = (*xpos2) * (*ypos);
				(*zpos3++) = (*xpos3) * (*ypos);
				ypos++;
			}
			xpos0 += 4;
			xpos1 += 4;
			xpos2 += 4;
			xpos3 += 4;

			zpos0 += 3 * cols;
			zpos1 += 3 * cols;
			zpos2 += 3 * cols;
			zpos3 += 3 * cols;
		}

		float *xpos = &_arr[r*4];
		float *zpos = &arr[r*4*cols];

		for (int i = r*4; i < rows; i++) {
			float *ypos = &p_tensor._arr[0];
			for (int j = 0; j < cols; j++) {
				(*zpos++) = (*xpos) * (*ypos++);
			}
			xpos++;
		}
	}
	else
	{
		float *xpos = &_arr[0];
		float *zpos = &arr[0];

		for (int i = 0; i < rows; i++) {
			float *ypos = &p_tensor._arr[0];
			for (int j = 0; j < cols; j++) {
				(*zpos++) = (*xpos) * (*ypos++);
			}
			xpos++;
		}
	}


	return Tensor(rank, shape, arr);
}

Tensor Tensor::apply(Tensor& p_source, float(*f)(float))
{
	float* arr = alloc_arr(p_source._size);
	int* shape = copy_shape(p_source._rank, p_source._shape);

	for (int i = 0; i < p_source._size; i++) {
		arr[i] = f(p_source._arr[i]);
	}

	return Tensor(p_source._rank, shape, arr);
}

Tensor Tensor::apply(Tensor& p_source1, Tensor& p_source2, float(*f)(float, float)) {
	float* arr = alloc_arr(p_source1._size);
	int* shape = copy_shape(p_source1._rank, p_source1._shape);

	for (int i = 0; i < p_source1._size; i++) {
		arr[i] = f(p_source1._arr[i], p_source2._arr[i]);
	}

	return Tensor(p_source1._rank, shape, arr);
}

int Tensor::max_value_index() const {
	int max = 0;

	for(int i = 0; i < _size; i++) {
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
	fill(VALUE, p_value);
}

float Tensor::sum() const {
	float s = 0;

	for (int i = 0; i < _size; i++) {
		s += _arr[i];
	}

	return s;
}


float Tensor::at(const int p_x) const {
	return _arr[p_x];
}

float Tensor::at(const int p_y, const int p_x) const {
	return _arr[p_y * _shape[1] + p_x];
}

float Tensor::at(const int p_z, const int p_y, const int p_x) const {
	return _arr[p_z * _shape[2] * _shape[1] + p_y * _shape[1] + p_x];
}

void Tensor::set(const int p_x, const float p_val) const {
	_arr[p_x] = p_val;
}

void Tensor::set(const int p_y, const int p_x, const float p_val) const {
	_arr[p_y * _shape[1] + p_x] = p_val;
}

void Tensor::set(const int p_z, const int p_y, const int p_x, const float p_val) const {
	_arr[p_z * _shape[2] * _shape[1] + p_y * _shape[1] + p_x] = p_val;
}

void Tensor::inc(const int p_x, const float p_val) const {
	_arr[p_x] += p_val;
}

void Tensor::inc(const int p_x, const int p_y, const float p_val) const {
	_arr[p_y * _shape[1] + p_x] += p_val;
}

void Tensor::dec(const int p_x, const float p_val) const {
	_arr[p_x] -= p_val;
}

void Tensor::dec(const int p_x, const int p_y, const float p_val) const {
	_arr[p_y * _shape[1] + p_x] -= p_val;
}

float* Tensor::alloc_arr(const int p_size) {
	float* res = static_cast<float*>(Alloc(p_size * sizeof(float)));
	return res;
	//return static_cast<float*>(malloc(p_size * sizeof(float)));
}

int* Tensor::alloc_shape(const int p_size) {
	return static_cast<int*>(Alloc(p_size * sizeof(int)));
	//return static_cast<int*>(malloc(p_size * sizeof(int)));
}

float Tensor::ew_dot(const float p_x, const float p_y) {
	return p_x * p_y;
}

float Tensor::ew_div(const float p_x, const float p_y) {
	return p_x / p_y;
}

float Tensor::ew_abs(const float p_x) {
	return abs(p_x);
}

float Tensor::sgn(const float p_x) {
	float res = 0;

	if (p_x > 0) res = 1;
	if (p_x < 0) res = -1;

	return res;
}

float Tensor::dist(Tensor* p_tensor1, Tensor* p_tensor2)
{
	float result = 0;

	for(int i = 0; i < p_tensor1->_size; i++)
	{
		result += std::pow(p_tensor1->_arr[i] - p_tensor2->_arr[i], 2);
	}

	return std::sqrt(result);
}

float* Tensor::dot(const Tensor* p_x, const Tensor* p_y)
{
	float* arr = alloc_arr(p_x->_size);

	float *xpos = &p_x->_arr[0];
	float *ypos = &p_y->_arr[0];
	float *zpos = &arr[0];

	for (int i = 0; i < p_x->_size; i++) {
		(*zpos++) = (*ypos++) * (*xpos++);
	}

	return arr;
}

float* Tensor::mat_vec(const Tensor *p_A, const Tensor *p_x)
{
	const int rows = p_A->_shape[0];
	const int cols = p_A->_shape[1];
	const int r = rows / 4;

	float *arr = alloc_arr(rows);

	float *Apos1 = &p_A->_arr[0];
	float *Apos2 = &p_A->_arr[cols * 1];
	float *Apos3 = &p_A->_arr[cols * 2];
	float *Apos4 = &p_A->_arr[cols * 3];
	float *ypos = &arr[0];

	if (rows > 3)
	{
		for (int i = 0; i < r; i++)
		{
			float ytemp1 = 0;
			float ytemp2 = 0;
			float ytemp3 = 0;
			float ytemp4 = 0;

			float *xpos = &p_x->_arr[0];

			for (int j = 0; j < cols; j++)
			{
				ytemp1 += (*Apos1++) * (*xpos);
				ytemp2 += (*Apos2++) * (*xpos);
				ytemp3 += (*Apos3++) * (*xpos);
				ytemp4 += (*Apos4++) * (*xpos);

				xpos++;
			}

			*ypos = ytemp1;
			ypos++;
			*ypos = ytemp2;
			ypos++;
			*ypos = ytemp3;
			ypos++;
			*ypos = ytemp4;
			ypos++;

			// skip next row
			Apos1 += 3 * cols;
			Apos2 += 3 * cols;
			Apos3 += 3 * cols;
			Apos4 += 3 * cols;
		}
	}

	if (rows % 4 != 0)
	{
		float *Apos = &p_A->_arr[r * 4 * cols];

		for (int i = r * 4; i < rows; i++) {
			arr[i] = 0;
			float *xpos = &p_x->_arr[0];
			for (int j = 0; j < cols; j++) {
				arr[i] += (*Apos++) * (*xpos++);
			}
		}
	}

	return arr;
}

void Tensor::free_arr() const {
	Free(_arr);
	//free(_arr);
}

void Tensor::free_shape() const {
	Free(_shape);
	//free(_shape);
}

int* Tensor::copy_shape(const int p_rank, int* p_shape) {
	int* shape = alloc_shape(p_rank);

	for (int i = 0; i < p_rank; i++) {
		shape[i] = p_shape[i];
	}

	return shape;
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
			for (int i = 0; i < _size; i++) _arr[i] = 0;
			break;
		case ONES:
			if (_rank == 1) {
				for (int i = 0; i < _size; i++) _arr[i] = 1;
			}
			if (_rank == 2 && _shape[0] == _shape[1]) {
				for (int i = 0; i < _shape[0]; i++) _arr[i * _shape[1] + i] = 1;
			}			
			break;
		case VALUE:
			for (int i = 0; i < _size; i++) _arr[i] = p_value;
			break;
		case RANDOM:
			for (int i = 0; i < _size; i++) _arr[i] = RandomGenerator::getInstance().random(-p_value, p_value);
			break;
	}
}

Tensor Tensor::concat(Tensor& p_tensor1, Tensor& p_tensor2) {
	float *res = alloc_arr(p_tensor1._size + p_tensor2._size);

	int index = 0;

	for (int i = 0; i < p_tensor1._size; i++) {
		res[index] = p_tensor1._arr[i];
		index++;
	}

	for (int i = 0; i < p_tensor2._size; i++) {
		res[index] = p_tensor2._arr[i];
		index++;
	}

	return Tensor({ p_tensor1._size + p_tensor2._size }, res);
}

void Tensor::concat(Tensor* p_result, Tensor* p_tensor1, Tensor* p_tensor2) {
	int index = 0;

	for (int i = 0; i < p_tensor1->_size; i++) {
		p_result->_arr[index] = p_tensor1->_arr[i];
		index++;
	}

	for (int i = 0; i < p_tensor2->_size; i++) {
		p_result->_arr[index] = p_tensor2->_arr[i];
		index++;
	}
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
		for(int j = 0; j < v.size(); j++)
		{
			res[i] = v[j];
			i++;
		}
	}

	return Tensor({ size }, res);
}

