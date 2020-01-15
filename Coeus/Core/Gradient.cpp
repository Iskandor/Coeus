#include "Gradient.h"

using namespace Coeus;

Gradient::Gradient()
= default;

Gradient::Gradient(map<string, Tensor>& p_buffer)
{
	for (auto& it : p_buffer)
	{
		_buffer[it.first] = p_buffer[it.first];
	}
}

Gradient::Gradient(Gradient& p_copy)
{
	for (auto& it : p_copy._buffer)
	{
		_buffer[it.first] = it.second;
	}
}


Gradient::~Gradient() = default;

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

bool Gradient::is_invalid()
{
	bool result = false;
	for (auto& it : _buffer)
	{
		if (it.second.has_NaN_Inf())
		{
			result = true;
		}
	}

	return result;
}

Tensor& Gradient::operator[](const string& p_id)
{
	return _buffer[p_id];
}

Gradient& Gradient::operator=(const Gradient& p_copy)
{
	for (auto& it : p_copy._buffer)
	{
		_buffer[it.first] = it.second;
	}
	return *this;
}

Gradient& Gradient::operator+=(const Gradient& p_rhs)
{
	for (auto& it : _buffer)
	{
		it.second += p_rhs._buffer.at(it.first);
	}
	return *this;
}

Gradient& Gradient::operator+=(const map<string, Tensor>& p_rhs)
{
	for (auto& it : _buffer)
	{
		it.second += p_rhs.at(it.first);
	}
	return *this;
}
