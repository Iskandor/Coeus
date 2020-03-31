#pragma once
#include "param.h"
#include <map>

class __declspec(dllexport) param_model
{
public:
	param_model();
	~param_model();

	param* add_param(std::initializer_list<int> p_shape);
	void add_model(param_model& p_model);

	std::map<std::string, param*>::iterator begin() { return _model.begin(); }
	std::map<std::string, param*>::iterator end() { return _model.end(); }

	std::map<std::string, tensor> zero_model();

private:
	std::map<std::string, param*> _model;
};

