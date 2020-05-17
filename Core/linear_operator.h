#pragma once
#include "igate.h"
#include "param.h"

class linear_operator : public igate
{
public:
	linear_operator(param* p_weights, param* p_bias);
	~linear_operator();

	tensor& forward(tensor& p_input) override;
	tensor& backward(tensor& p_delta) override;

private:
	tensor*	_input;
	tensor	_output;
	tensor	_delta;
	param* _weights;
	param* _bias;

	bool _gpu_flag;
};

