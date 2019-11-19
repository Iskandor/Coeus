#include "TensorPool.h"

TensorPool::TensorPool()
= default;


TensorPool& TensorPool::instance()
{
	static TensorPool instance;
	return instance;
}

TensorPool::~TensorPool()
{
	for(auto it: _pool)
	{
		while(!it.second.empty())
		{
			delete it.second.top();
			it.second.pop();
		}		
	}

	_pool.clear();
}

float* TensorPool::get(const int p_size)
{
	float* result = nullptr;

	if (_pool.find(p_size) != _pool.end())
	{
		if (_pool[p_size].empty())
		{
			result = static_cast<float*>(malloc(p_size * sizeof(float)));
		}
		else
		{
			result = _pool[p_size].top();
			_pool[p_size].pop();
		}
	}
	else
	{
		result = static_cast<float*>(malloc(p_size * sizeof(float)));
	}

	return result;
}

void TensorPool::release(const int p_size, float* p_buffer)
{
	_pool[p_size].push(p_buffer);
}
