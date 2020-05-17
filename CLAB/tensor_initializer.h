#pragma once
#include "tensor.h"

class COEUS_DLL_API tensor_initializer
{
public:
	enum TYPE
	{
		DEBUG,
		UNIFORM,
		NORMAL,
		LECUN_UNIFORM,
		LECUN_NORMAL,
		GLOROT_UNIFORM,
		GLOROT_NORMAL,
		XAVIER_UNIFORM,
		XAVIER_NORMAL
	};

	tensor_initializer(tensor_initializer& p_copy);

	static tensor_initializer* uniform(float p_lower_bound, float p_upper_bound);
	static tensor_initializer* normal(float p_mean, float p_sigma);
	static tensor_initializer* lecun_uniform();
	static tensor_initializer* lecun_normal();
	static tensor_initializer* glorot_uniform();
	static tensor_initializer* glorot_normal();
	static tensor_initializer* xavier_uniform();
	static tensor_initializer* xavier_normal();
	static tensor_initializer* debug(float p_value);
	
	~tensor_initializer();

	void init(tensor& p_tensor) const;

private:
	tensor_initializer(TYPE p_type, float p_arg1, float p_arg2);

	void static init_uniform(tensor& p_tensor, float p_lower_bound, float p_upper_bound);
	void static init_normal(tensor& p_tensor, float p_mean, float p_sigma);

	TYPE	_type;
	float	_arg1;
	float	_arg2;
};

