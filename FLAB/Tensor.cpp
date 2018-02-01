#include "Tensor.h"
#include <cstdarg>
#include <cstdlib>
#include "RandomGenerator.h"
#include <ppl.h>
//#include "TensorPool.h"

using namespace FLAB;
using namespace Concurrency;

Tensor::Tensor(): _arr(nullptr), _rank(0), _shape(nullptr), _size(0) {
}

Tensor::Tensor(const initializer_list<int> p_shape, INIT p_init, double p_value) {
	init_shape(p_shape);
	_arr = alloc_arr(_size);
	fill(p_init, p_value);
}

Tensor::Tensor(const initializer_list<int> p_shape, double* p_data) {
	init_shape(p_shape);
	_arr = p_data;
}

Tensor::Tensor(int p_rank, int* p_shape, double* p_data) {
	init_shape(p_rank, p_shape);
	_arr = p_data;
}

Tensor::Tensor(const initializer_list<int> p_shape, initializer_list<double> p_inputs) {
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

	for(int i = 0; i < _rank; i++) {
		_shape[i] = p_copy._shape[i];
	}
	
	_arr = alloc_arr(_size);
	memcpy(_arr, p_copy._arr, sizeof(double) * static_cast<size_t>(_size));
}

Tensor::~Tensor() {
	if (_shape != nullptr) free_shape();
	_shape = nullptr;
	if (_arr != nullptr) free_arr();
	_arr = nullptr;
}

Tensor Tensor::Zero(const initializer_list<int> p_shape) {
	return Tensor(p_shape, ZERO);
}

Tensor Tensor::Ones(const initializer_list<int> p_shape) {
	return Tensor(p_shape, ONES);
}

Tensor Tensor::Value(const initializer_list<int> p_shape, double p_value) {
	return Tensor(p_shape, VALUE, p_value);
}

Tensor Tensor::Random(const initializer_list<int> p_shape, double p_limit) {
	return Tensor(p_shape, RANDOM, p_limit);
}

void Tensor::operator=(const Tensor& p_copy) {
	if (_shape != nullptr) {
		free_shape();
	}

	if (_arr != nullptr) {
		free_arr();
	}

	_rank = p_copy._rank;
	_shape = alloc_shape(_rank);
	_size = p_copy._size;

	for (int i = 0; i < _rank; i++) {
		_shape[i] = p_copy._shape[i];
	}

	_arr = alloc_arr(_size);
	memcpy(_arr, p_copy._arr, sizeof(double) * static_cast<size_t>(_size));
}

Tensor Tensor::operator+(const Tensor& p_tensor) {
	double* arr = alloc_arr(_size);

	for(int i = 0; i < _size; i++) {
		arr[i] = _arr[i] + p_tensor._arr[i];
	}

	return Tensor(_rank, _shape, arr);
}

void Tensor::operator+=(const Tensor& p_tensor) {
	for (int i = 0; i < _size; i++) {
		_arr[i] += p_tensor._arr[i];
	}
}

Tensor Tensor::operator-(const Tensor& p_tensor) {
	double* arr = alloc_arr(_size);

	for (int i = 0; i < _size; i++) {
		arr[i] = _arr[i] - p_tensor._arr[i];
	}

	return Tensor(_rank, _shape, arr);
}

void Tensor::operator-=(const Tensor& p_tensor) {
	for (int i = 0; i < _size; i++) {
		_arr[i] -= p_tensor._arr[i];
	}
}

Tensor Tensor::operator*(const Tensor& p_tensor) {
	double* arr = nullptr;
	int rank = 0;
	int* shape = nullptr;
	
	if (this->_rank == 1 && p_tensor._rank == 1) {
		arr = alloc_arr(_size * p_tensor._size);
		rank = 2;
		shape = alloc_shape(rank);
		shape[0] = _size;
		shape[1] = p_tensor._size;

		for (int i = 0; i < this->_size; i++) {
			for (int j = 0; j < p_tensor._size; j++) {
				arr[i * p_tensor._size + j] = _arr[i] * p_tensor._arr[j];
			}
		}
	}

	if (this->_rank == 2 && p_tensor._rank == 1) {
		arr = alloc_arr(p_tensor._size);
		rank = 1;
		shape = alloc_shape(rank);
		shape[0] = p_tensor._size;

		for (int i = 0; i < _shape[0]; i++) {
			for (int j = 0; j < _shape[1]; j++) {
				arr[i] = arr[i] + _arr[i * _shape[1] + j] * p_tensor._arr[j];
			}
		}
	}

	if (this->_rank == 2 && p_tensor._rank == 2) {
		arr = alloc_arr(_shape[0] * p_tensor._shape[1]);
		rank = 2;
		shape = alloc_shape(rank);
		shape[0] = _shape[0];
		shape[1] = p_tensor._shape[1];

		for (int i = 0; i < _shape[0]; i++) {
			for (int j = 0; j < p_tensor._shape[1]; j++) {
				for (int k = 0; k < _shape[1]; k++) {
					arr[i * _shape[1] + j] += _arr[i * _shape[1] + k] * p_tensor._arr[k * p_tensor._shape[1] + j];
				}
			}
		}
	}

	return Tensor(rank, shape, arr);
}

Tensor Tensor::operator*(const double p_const) {
	double* arr = alloc_arr(_size);

	for (int i = 0; i < _size; i++) {
		arr[i] = _arr[i] * p_const;
	}

	return Tensor(_rank, _shape, arr);
}

void Tensor::operator*=(const double p_const) {
	for (int i = 0; i < _size; i++) {
		_arr[i] *= p_const;
	}
}

Tensor Tensor::apply(double(*f)(double)) const 
{
	double* arr = alloc_arr(_size);

	for (int i = 0; i < _size; i++) {
		arr[i] = f(_arr[i]);
	}

	return Tensor(_rank, _shape, arr);
}

int Tensor::max_index() const {
	int max = 0;

	for(int i = 0; i < _size; i++) {
		if (_arr[max] < _arr[i]) max = i;
	}

	return max;
}

void Tensor::override(Tensor* p_tensor) const {
	memcpy(_arr, p_tensor->_arr, sizeof(double) * static_cast<size_t>(_size));
}

void Tensor::fill(const double p_value) const {
	fill(VALUE, p_value);
}

double Tensor::sum() const {
	double s = 0;

	for (int i = 0; i < _size; i++) {
		s += _arr[i];
	}

	return s;
}


double Tensor::at(int p_x) const {
	return _arr[p_x];
}

double Tensor::at(int p_y, int p_x) const {
	return _arr[p_y * _shape[1] + p_x];
}

void Tensor::set(int p_x, double p_val) const {
	_arr[p_x] = p_val;
}

void Tensor::set(int p_y, int p_x, double p_val) const {
	_arr[p_y * _shape[1] + p_x] = p_val;
}

double* Tensor::alloc_arr(const int p_size) {
	return static_cast<double*>(Alloc(p_size * sizeof(double)));
	//return static_cast<double*>(malloc(p_size * sizeof(double)));
}

int* Tensor::alloc_shape(const int p_size) {
	return static_cast<int*>(Alloc(p_size * sizeof(int)));
	//return static_cast<int*>(malloc(p_size * sizeof(int)));
}

void Tensor::free_arr() const {
	Free(_arr);
	//free(_arr);
}

void Tensor::free_shape() const {
	Free(_shape);
	//free(_shape);
}

void Tensor::init_shape(const int p_rank, int* p_shape) {
	_rank = p_rank;
	_shape = alloc_shape(_rank);
	_size = 1;

	for(int i = 0; i < _rank; i++) {
		_shape[i] = p_shape[i];
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

void Tensor::fill(const INIT p_init, const double p_value) const {
	switch (p_init) {
		case ZERO:
			for (int i = 0; i < _size; i++) _arr[i] = 0;
			break;
		case ONES:
			if (_rank == 1) {
				for (int i = 0; i < _size; i++) _arr[i] = 1;
			}
			if (_rank == 2) {
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

Tensor Tensor::Concat(Tensor& p_tensor1, Tensor& p_tensor2) {
	double *res = alloc_arr(p_tensor1._size + p_tensor2._size);

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

void Tensor::Concat(Tensor* p_result, Tensor* p_tensor1, Tensor* p_tensor2) {
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
