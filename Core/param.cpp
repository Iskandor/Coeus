#include "param.h"
#include "id_generator.h"

param::param(const std::initializer_list<int> p_shape)
{
	_id = id_generator::next();
	_params = tensor(p_shape);
	_gradient = tensor(p_shape);
}

param::param(param& p_copy)
{
	_id = p_copy._id;
	_params = p_copy._params;
	_gradient = p_copy._gradient;
}

param::~param()
= default;
