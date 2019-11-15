#include "Gradient.h"

using namespace Coeus;

Gradient::Gradient()
{
}

Gradient::Gradient(map<string, Tensor>& p_buffer)
{
	for (auto& it : p_buffer)
	{
		_buffer[it.first] = p_buffer[it.first];
	}
}


Gradient::~Gradient()
{
}

void Gradient::init(ParamModel* p_model)
{
	_buffer = p_model->get_empty_params();
}

void Gradient::fill(const float p_value)
{
	for (auto& it : _buffer)
	{
		it.second.fill(p_value);
	}
}

Tensor& Gradient::operator[](const string& p_id)
{
	return _buffer[p_id];
}

Gradient& Gradient::operator+=(const Gradient& p_rhs)
{
	for (auto& it : _buffer)
	{
		it.second += p_rhs._buffer.at(it.first);
	}
	return *this;
}