#pragma once
#include "coeus.h"
#include "tensor.h"

class COEUS_DLL_API igate
{
public:
	igate() = default;
	virtual ~igate() = default;

	virtual tensor& forward(tensor& p_input) = 0;
	virtual tensor& backward(tensor& p_delta) = 0;

};
