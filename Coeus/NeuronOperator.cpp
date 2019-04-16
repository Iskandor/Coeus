#include "Coeus.h"
#include "NeuronOperator.h"
#include "ActivationFunctionFactory.h"
#include "IDGen.h"
#include "TensorOperator.h"
#include "IOUtils.h"


using namespace Coeus;

NeuronOperator::NeuronOperator(const int p_dim, const ACTIVATION p_activation)
{
	_id = IDGen::instance().next();
	_dim = p_dim;
	_bias = new Param(IDGen::instance().next(), new Tensor({ p_dim }, Tensor::ZERO));
	add_param(_bias->get_id(), _bias->get_data());

	_activation_function = ActivationFunctionFactory::create_function(p_activation);
	_net = nullptr;
	_dnet = nullptr;
	_int = nullptr;
	_output = nullptr;
}

NeuronOperator::NeuronOperator(json p_data)
{
	_id = p_data["id"].get<string>();
	_dim = p_data["dim"].get<int>();
	_activation_function = IOUtils::init_activation_function(p_data["f"]);
	_bias = IOUtils::load_param(p_data["b"]);
	add_param(_bias->get_id(), _bias->get_data());

	_net = nullptr;
	_dnet = nullptr;
	_int = nullptr;
	_output = nullptr;
}

NeuronOperator::NeuronOperator(NeuronOperator& p_copy) : ParamModel(p_copy)
{
	_id = p_copy._id;
	_dim = p_copy._dim;
	_bias = new Param(*p_copy._bias);
	add_param(_bias->get_id(), _bias->get_data());

	_activation_function = ActivationFunctionFactory::create_function(p_copy._activation_function->get_type());
	_net = nullptr;
	_dnet = nullptr;
	_int = nullptr;
	_output = nullptr;
}

NeuronOperator::~NeuronOperator()
{
	delete _bias;

	delete _activation_function;
	delete _net;
	delete _dnet;
	delete _int;
	delete _output;
}

void NeuronOperator::integrate(Tensor* p_input, Tensor* p_weights)
{
	if (p_input->rank() == 1)
	{
		_net = init_auxiliary_parameter(_net, 1, _dim);
		_int = init_auxiliary_parameter(_int, 1, _dim);
		TensorOperator::instance().full_int_s(_int->arr(), p_input->arr(), p_weights->arr(), p_weights->shape(0), p_weights->shape(1));
		TensorOperator::instance().vv_add(_net->arr(), _int->arr(), _net->arr(), _dim);
	}
	if (p_input->rank() == 2)
	{
		_net = init_auxiliary_parameter(_net, p_input->shape(0), _dim);
		_int = init_auxiliary_parameter(_int, p_input->shape(0), _dim);
		TensorOperator::instance().full_int_b(p_input->shape(0), _int->arr(), p_input->arr(), p_weights->arr(), p_weights->shape(0), p_weights->shape(1));
		TensorOperator::instance().vv_add(_net->arr(), _int->arr(), _net->arr(), p_input->shape(0) * _dim);
	}
}

void NeuronOperator::activate()
{
	if (_net->rank() == 1)
	{
		TensorOperator::instance().full_bias_s(_net->arr(), _bias->get_data()->arr(), _dim);
		_output = init_auxiliary_parameter(_output, 1, _dim);
		_dnet = init_auxiliary_parameter(_dnet, 1, _dim);
	}
	if (_net->rank() == 2)
	{
		TensorOperator::instance().full_bias_b(_net->shape(0), _net->arr(), _bias->get_data()->arr(), _dim);
		_output = init_auxiliary_parameter(_output, _net->shape(0), _dim);
		_dnet = init_auxiliary_parameter(_dnet, _net->shape(0), _dim);
	}

	*_output = _activation_function->activate(*_net);
	_dnet->override(_net);
	_net->fill(0);
}

Tensor NeuronOperator::derivative() const
{
	return _activation_function->derivative(*_dnet);
}

json NeuronOperator::get_json() const
{
	json data;

	data["id"] = _id;
	data["dim"] = _dim;
	data["f"] = _activation_function->get_json();
	data["b"] = IOUtils::save_param(_bias);

	return data;
}

Tensor* NeuronOperator::init_auxiliary_parameter(Tensor* p_param, const int p_rows, const int p_cols)
{
	if (p_rows == 1)
	{
		if (p_param == nullptr || p_param->rank() != 1)
		{
			delete p_param;
			p_param = new Tensor({ p_cols }, Tensor::ZERO);
		}
	}
	if (p_rows > 1)
	{
		if (p_param == nullptr || p_param->rank() != 2)
		{
			delete p_param;
			p_param = new Tensor({ p_rows, p_cols }, Tensor::ZERO);
		}
	}

	return p_param;
}
