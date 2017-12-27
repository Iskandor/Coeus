#include "TensorPool.h"

using namespace FLAB;

TensorPool& TensorPool::instance() {
	static TensorPool instance;
	return instance;
}

double* TensorPool::get(const unsigned int p_size) {
	double* result = nullptr;

	if(_pool.find(p_size) == _pool.end()) {
		_pool[p_size] = new stack<double*>();
		result = static_cast<double*>(calloc(static_cast<size_t>(p_size), sizeof(double)));
	}
	else {
		if (_pool[p_size]->empty()) {
			result = static_cast<double*>(calloc(static_cast<size_t>(p_size), sizeof(double)));
		}
		else {
			result = _pool[p_size]->top();
			_pool[p_size]->pop();
		}
	}

	_counter++;

	return result;
}

void TensorPool::release(double* p_buffer, unsigned int p_size) {
	_pool[p_size]->push(p_buffer);
	_counter--;
}

TensorPool::TensorPool()
{
	_counter = 0;
}


TensorPool::~TensorPool()
{
	for(auto it = _pool.begin(); it != _pool.end(); ++it) {
		while(!(*it).second->empty()) {
			free((*it).second->top());
			(*it).second->pop();
		}
		delete (*it).second;
	}
}
