#pragma once
#include "tensor.h"

class COEUS_DLL_API variable
{
public:
	variable();
	~variable();

	tensor& value() { return _value; }
	tensor& delta() { return _delta; }

private:
	tensor _value;
	tensor _delta;
};

