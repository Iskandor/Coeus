#include "TensorPool.h"

using namespace FLAB;

TensorPool* TensorPool::getInstance() {
	if (_instance == nullptr) {
		_instance = new TensorPool();
	}

	return _instance;
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

	return result;
}

void TensorPool::release(double* p_buffer) {
	const unsigned int size = sizeof(p_buffer) / sizeof(double);

	_pool[size]->push(p_buffer);

}

TensorPool::TensorPool()
{
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
