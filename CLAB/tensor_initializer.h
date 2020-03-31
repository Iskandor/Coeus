#pragma once
#include "tensor.h"

class __declspec(dllexport) tensor_initializer
{
public:
	enum TYPE
	{
		UNIFORM,
		NORMAL,
		LECUN_UNIFORM,
		LECUN_NORMAL,
		GLOROT_UNIFORM,
		GLOROT_NORMAL
	};

	static tensor_initializer* uniform(float p_lower_bound, float p_upper_bound);
	static tensor_initializer* normal(float p_mean, float p_sigma);
	static tensor_initializer* lecun_uniform();
	static tensor_initializer* lecun_normal();
	static tensor_initializer* glorot_uniform();
	static tensor_initializer* glorot_normal();
	
	~tensor_initializer();

	void init(tensor& p_tensor);

private:
	tensor_initializer(TYPE p_type, float p_arg1, float p_arg2);

	void static init_uniform(tensor& p_tensor, float p_lower_bound, float p_upper_bound);
	void static init_normal(tensor& p_tensor, float p_mean, float p_sigma);

	TYPE	_type;
	float	_arg1;
	float	_arg2;
};

