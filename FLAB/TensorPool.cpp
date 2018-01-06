#include "TensorPool.h"

using namespace FLAB;

TensorPool& TensorPool::instance() {
	static TensorPool instance;
	return instance;
}

double* TensorPool::get_dbl(const unsigned int p_size) {
	double* result = nullptr;

	if(_pool_dbl.find(p_size) == _pool_dbl.end()) {
		_pool_dbl[p_size] = new stack<double*>();
		result = static_cast<double*>(calloc(static_cast<size_t>(p_size), sizeof(double)));
	}
	else {
		if (_pool_dbl[p_size]->empty()) {
			result = static_cast<double*>(calloc(static_cast<size_t>(p_size), sizeof(double)));
		}
		else {
			result = _pool_dbl[p_size]->top();
			_pool_dbl[p_size]->pop();
		}
	}

	_counter++;

	return result;
}

int* TensorPool::get_int(const unsigned int p_size) {
	int* result = nullptr;

	if (_pool_int.find(p_size) == _pool_int.end()) {
		_pool_int[p_size] = new stack<int*>();
		result = static_cast<int*>(calloc(static_cast<size_t>(p_size), sizeof(int)));
	}
	else {
		if (_pool_int[p_size]->empty()) {
			result = static_cast<int*>(calloc(static_cast<size_t>(p_size), sizeof(int)));
		}
		else {
			result = _pool_int[p_size]->top();
			_pool_int[p_size]->pop();
		}
	}

	_counter++;

	return result;
}

void TensorPool::release(double* p_buffer, const unsigned p_size) {
	_pool_dbl[p_size]->push(p_buffer);
	_counter--;
}

void TensorPool::release(int* p_buffer, const unsigned p_size) {
	_pool_int[p_size]->push(p_buffer);
	_counter--;
}

TensorPool::TensorPool()
{
	_counter = 0;
}


TensorPool::~TensorPool()
{
	for(auto it = _pool_dbl.begin(); it != _pool_dbl.end(); ++it) {
		while(!(*it).second->empty()) {
			free((*it).second->top());
			(*it).second->pop();
		}
		delete (*it).second;
	}

	for (auto it = _pool_int.begin(); it != _pool_int.end(); ++it) {
		while (!(*it).second->empty()) {
			free((*it).second->top());
			(*it).second->pop();
		}
		delete (*it).second;
	}
}
