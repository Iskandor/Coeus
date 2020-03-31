#pragma once
#include <string>
#include "tensor.h"

class param
{
public:
	param(std::initializer_list<int> p_shape);
	~param();

	std::string& id() { return _id; }
	tensor& params() { return _params; }
	tensor& gradient() { return _gradient; }

private:
	std::string _id;
	tensor		_params;
	tensor		_gradient;

};

