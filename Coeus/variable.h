#pragma once
#include "tensor.h"

class __declspec(dllexport) variable
{
public:
	variable();
	~variable();

	void resize(std::initializer_list<int> p_shape);
	tensor& value() { return _value; }
	tensor& delta() { return _delta; }

private:
	tensor _value;
	tensor _delta;
};

