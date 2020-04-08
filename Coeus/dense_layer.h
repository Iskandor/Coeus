#pragma once
#include "linear_operator.h"
#include "activation_functions.h"
#include <vector>
#include "param_model.h"

class __declspec(dllexport) dense_layer : public igate
{
public:
	dense_layer(std::string p_id, int p_dim, activation_function* p_activation_function, tensor_initializer* p_initializer, std::initializer_list<int> p_input_shape = {});
	dense_layer(dense_layer& p_copy);
	~dense_layer();

	tensor& forward(tensor& p_input) override;
	tensor& backward(tensor& p_delta) override;

	void init(param_model* p_model, std::vector<dense_layer*>& p_input_layers);

	std::string& id() { return _id; }
	int dim() const { return _dim; }
	int input_dim() const { return _input_dim; }

	tensor* output() { return &_output; }

private:
	int get_input_dim(std::initializer_list<int> &p_input_shape) const;

	std::string _id;
	int			_dim;
	int			_input_dim;
	tensor		_output;

	param* _weights;
	param* _bias;
	tensor_initializer*	_initializer;

	linear_operator*		_op;
	activation_function*	_af;
};

