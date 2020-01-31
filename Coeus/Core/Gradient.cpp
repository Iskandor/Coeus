#include "Gradient.h"

using namespace Coeus;

Gradient::Gradient(): _params_size(0)
{
}

Gradient::Gradient(map<string, Tensor>& p_buffer)
{
	_params_size = 0;
	for (auto& it : p_buffer)
	{
		_buffer[it.first] = p_buffer[it.first];
		_params_size += it.second.size();
	}
	if (_gradient.size() != _params_size) _gradient = Tensor({ _params_size }, Tensor::ZERO);
}

Gradient::Gradient(Gradient& p_copy)
{
	_params_size = p_copy._params_size;
	for (auto& it : p_copy._buffer)
	{
		_buffer[it.first] = it.second;
	}
	if (_gradient.size() != _params_size) _gradient = Tensor({ _params_size }, Tensor::ZERO);
}


Gradient::~Gradient() = default;

void Gradient::init(ParamModel* p_model)
{
	_buffer = p_model->get_empty_params();
	_params_size = p_model->get_params_size();
	if (_gradient.size() != _params_size) _gradient = Tensor({ _params_size }, Tensor::ZERO);
}

void Gradient::fill(const float p_value)
{
	for (auto& it : _buffer)
	{
		it.second.fill(p_value);
	}
}

void Gradient::fill(Tensor& p_gradient)
{
	_gradient.override(&p_gradient);
	reshape();
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
	_params_size = p_copy._params_size;
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

Gradient& Gradient::operator/=(const float p_rhs)
{
	for (auto& it : _buffer)
	{
		it.second /= p_rhs;
	}
	return *this;
}

void Gradient::flatten()
{
	if (_gradient.size() != _params_size) _gradient = Tensor({ _params_size }, Tensor::ZERO);

	_gradient.reset_index();
	for (auto& it : _buffer)
	{
		_gradient.push_back(&it.second);
	}
}

void Gradient::reshape()
{
	int start_index = 0;
	for (auto& it : _buffer)
	{
		_gradient.splice(start_index, &it.second);
		start_index += it.second.size();
	}
}
