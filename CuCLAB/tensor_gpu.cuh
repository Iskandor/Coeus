#pragma once
class __declspec(dllexport) tensor_gpu
{
	friend class tensor;
private:
	tensor_gpu(int p_size);
	tensor_gpu(const tensor_gpu& p_copy);
	tensor_gpu& operator=(const tensor_gpu& p_copy);
	~tensor_gpu();

	void to_gpu(float* p_cpu_data) const;
	void to_cpu(float* p_cpu_data) const;

	static float* init_data(int &p_size);

	int		_size;
	float*	_data;
};

