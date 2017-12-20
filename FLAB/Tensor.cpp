#include "Tensor.h"
#include <cstdarg>
#include <cstdlib>
#include "RandomGenerator.h"

using namespace FLAB;

Tensor::Tensor(): _rank(0), _shape(nullptr), _size(0) {
}

Tensor::Tensor(const initializer_list<int> p_shape, INIT p_init, double p_value) {
	init_shape(p_shape);
	_arr = static_cast<double*>(calloc(_size, sizeof(double)));
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
	_arr = static_cast<double*>(calloc(_size, sizeof(double)));

	int i = 0;
	for(auto v = p_inputs.begin(); v != p_inputs.end(); v++) {
		_arr[i] = *v;
		i++;
	}
}

Tensor::Tensor(const Tensor& p_copy) {
	_rank = p_copy._rank;
	_shape = new int[_rank];
	_size = p_copy._size;

	for(int i = 0; i < _rank; i++) {
		_shape[i] = p_copy._shape[i];
	}
	
	_arr = p_copy._arr;
}

Tensor::~Tensor() {
	if (_shape != nullptr) delete _shape;
	if (_arr != nullptr) delete _arr;
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
	_rank = p_copy._rank;
	_shape = new int[_rank];
	_size = p_copy._size;

	for (int i = 0; i < _rank; i++) {
		_shape[i] = p_copy._shape[i];
	}

	_arr = static_cast<double*>(calloc(_size, sizeof(double)));
	memcpy(_arr, p_copy._arr, sizeof(double) * static_cast<size_t>(_size));
}

Tensor Tensor::operator+(const Tensor& p_tensor) {
	double* arr = new double[_size];

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
	double* arr = new double[_size];

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
		arr = new double[this->_size * p_tensor._size];
		rank = 2;
		shape = new int[rank];
		shape[0] = _size;
		shape[1] = p_tensor._size;

		for (int i = 0; i < this->_size; i++) {
			for (int j = 0; j < p_tensor._size; j++) {
				arr[i * p_tensor._size + j] = _arr[i] * p_tensor._arr[j];
			}
		}
	}

	if (this->_rank == 2 && p_tensor._rank == 1) {
		arr = new double[p_tensor._size];
		rank = 1;
		shape = new int[rank];
		shape[0] = p_tensor._size;

		for (int i = 0; i < _shape[0]; i++) {
			for (int j = 0; j < _shape[1]; j++) {
				arr[i] = arr[i] + _arr[i * _shape[1] + j] * p_tensor._arr[j];
			}
		}
	}

	if (this->_rank == 2 && p_tensor._rank == 2) {
		arr = new double[_shape[0] * p_tensor._shape[1]];
		rank = 2;
		shape = new int[rank];
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
	double* arr = new double[_size];

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

Tensor Tensor::apply(double(*f)(double))
{
	double* arr = new double[_size];

	for (int i = 0; i < _size; i++) {
		arr[i] = f(_arr[i]);
	}

	return Tensor(_rank, _shape, arr);
}

void Tensor::fill(double p_value) {
	fill(VALUE, p_value);
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


void Tensor::init_shape(int p_rank, int* p_shape) {
	_rank = p_rank;
	_shape = new int[_rank];
	_size = 1;

	for(int i = 0; i < _rank; i++) {
		_shape[i] = p_shape[i];
		_size *= _shape[i];
	}
}

void Tensor::init_shape(initializer_list<int> p_shape) {
	_rank = p_shape.size();
	_shape = new int[_rank];
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
