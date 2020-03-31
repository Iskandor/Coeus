#include "variable.h"



variable::variable()
= default;


variable::~variable()
= default;

void variable::resize(const std::initializer_list<int> p_shape)
{
	_value.resize(p_shape);
	_delta.resize(p_shape);
}
