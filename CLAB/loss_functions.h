#pragma once
#include "igate.h"

class __declspec(dllexport) loss_function
{
public:
	loss_function() = default;
	virtual ~loss_function() = default;

	virtual float forward(tensor& p_prediction, tensor& p_target) = 0;
	virtual tensor& backward(tensor& p_prediction, tensor& p_target) = 0;

protected:
	tensor gradient;
	const int segment = 8;
};

class __declspec(dllexport) mse_function : public loss_function
{
public:
	mse_function() = default;
	~mse_function() = default;

	float forward(tensor& p_prediction, tensor& p_target) override;
	tensor& backward(tensor& p_prediction, tensor& p_target) override;
};
