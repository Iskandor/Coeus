#include "param.h"
#include "id_generator.h"

param::param(const std::initializer_list<int> p_shape)
{
	_id = id_generator::next();
	_params = tensor(p_shape);
	_gradient = tensor(p_shape);
}

param::~param()
= default;
