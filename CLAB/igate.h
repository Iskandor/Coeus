#pragma once
#include "tensor.h"

class __declspec(dllexport) igate
{
public:
	igate() = default;
	virtual ~igate() = default;

	virtual tensor& forward(tensor& p_input) = 0;
	virtual tensor& backward(tensor& p_delta) = 0;

};
