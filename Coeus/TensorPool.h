#pragma once
#include <map>
#include <stack>

using namespace std;

class __declspec(dllexport) TensorPool
{
public:
	static TensorPool& instance();
	TensorPool(TensorPool const&) = delete;
	void operator=(TensorPool const&) = delete;
	~TensorPool();

	float* get(int p_size);
	void release(int p_size, float* p_buffer);

private:
	TensorPool();

	map<int, stack<float*>> _pool;
};