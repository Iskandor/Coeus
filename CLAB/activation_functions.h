#pragma once
#include "igate.h"

class __declspec(dllexport) activation_function : public igate
{
public:
	enum TYPE
	{
		LINEAR = 0,
		SIGMOID = 1,
		TANH = 2,
		TANHEXP = 3
	};

	static activation_function* create(TYPE p_type);

	static activation_function* linear();
	static activation_function* sigmoid();
	static activation_function* tanh();
	static activation_function* tanhexp();
	~activation_function() = default;

	TYPE type() const { return _type; }

protected:
	activation_function() = default;	

	TYPE	_type;
	tensor	_input;
	
};

class __declspec(dllexport) linear_function : public activation_function
{
public:
	linear_function() { _type = LINEAR; }
	~linear_function() = default;

	tensor& forward(tensor& p_input) override;
	tensor& backward(tensor& p_delta) override;
};

class __declspec(dllexport) sigmoid_function : public activation_function
{
public:
	sigmoid_function() { _type = SIGMOID; }
	~sigmoid_function() = default;

	tensor& forward(tensor& p_input) override;
	tensor& backward(tensor& p_delta) override;
};

class __declspec(dllexport) tanh_function : public activation_function
{
public:
	tanh_function() { _type = TANH; }
	~tanh_function() = default;

	tensor& forward(tensor& p_input) override;
	tensor& backward(tensor& p_delta) override;
};

class __declspec(dllexport) tanhexp_function : public activation_function
{
public:
	tanhexp_function() { _type = TANHEXP; }
	~tanhexp_function() = default;

	tensor& forward(tensor& p_input) override;
	tensor& backward(tensor& p_delta) override;
};