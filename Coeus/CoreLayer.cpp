#include "CoreLayer.h"
#include "IDGen.h"
#include "TensorOperator.h"
#include "ActivationFunctionFactory.h"
#include "TensorInitializer.h"

using namespace Coeus;

CoreLayer::CoreLayer(const string& p_id, const int p_dim, const ACTIVATION p_activation, TensorInitializer* p_initializer, const int p_in_dim) : BaseLayer(p_id, p_dim, p_in_dim)
{
	_type = CORE;
	_y = new NeuronOperator(p_dim, p_activation);
	add_param(_y);
	_initializer = p_initializer;
	_W = nullptr;
}

CoreLayer::CoreLayer(const json& p_data) : BaseLayer(p_data)
{
	_type = CORE;
}

CoreLayer::CoreLayer(CoreLayer &p_copy) : BaseLayer(p_copy._id, p_copy._dim, p_copy._in_dim) {
	_type = CORE;
	_y = new NeuronOperator(*p_copy._y);
	add_param(_y);
	_initializer = p_copy._initializer;
	_W = nullptr;
}

CoreLayer::~CoreLayer()
{
	delete _y;
	delete _W;
	delete _initializer;
}

void CoreLayer::activate()
{
	_input->reset_index();
	_y->integrate(_input, _W->get_data());
	_y->activate();
	_output = _y->get_output();
}

void CoreLayer::calc_delta(map<string, Tensor*>& p_delta_map, map<string, Tensor*>& p_derivative_map)
{
	_in_derivative->reset_index();
	for (auto it : _input_layer)
	{
		_in_derivative->push_back(p_derivative_map[it->get_id()]);
	}

	Tensor*	 delta_out = p_delta_map[_id];
	Tensor*	 delta_in = nullptr;

	if (!_input_layer.empty())
	{
		delta_in = NeuronOperator::init_auxiliary_parameter(delta_in, _batch_size, _in_dim);

		if (_batch)
		{
			TensorOperator::instance().full_delta_b(_batch_size, delta_in->arr(), delta_out->arr(), _W->get_data()->arr(), _in_derivative->arr(), _W->get_data()->shape(0), _W->get_data()->shape(1));
		}
		else
		{
			TensorOperator::instance().full_delta_s(delta_in->arr(), delta_out->arr(), _W->get_data()->arr(), _in_derivative->arr(), _dim, _in_dim);
		}

		int index = 0;

		for (auto it : _input_layer)
		{
			p_delta_map[it->get_id()] = NeuronOperator::init_auxiliary_parameter(p_delta_map[it->get_id()], _batch_size, it->get_dim());
			delta_in->splice(index, p_delta_map[it->get_id()]);
			index += it->get_dim();
		}

		delete delta_in;
	}
}

void CoreLayer::calc_gradient(map<string, Tensor>& p_gradient_map, map<string, Tensor*>& p_delta_map, map<string, Tensor*>& p_derivative_map)
{
	Tensor* dW = &p_gradient_map[_W->get_id()];
	Tensor* delta = p_delta_map[_id];

	if (_batch)
	{
		TensorOperator::instance().m_reduce(p_gradient_map[_y->get_bias()->get_id()].arr(), delta->arr(), delta->shape(0), delta->shape(1));
		TensorOperator::instance().full_gradient_b(_batch_size, _input->arr(), delta->arr(), dW->arr(), _dim, _in_dim);
	}
	else
	{
		p_gradient_map[_y->get_bias()->get_id()].override(delta);
		TensorOperator::instance().full_gradient_s(_input->arr(), delta->arr(), dW->arr(), _dim, _in_dim);
	}
}

void CoreLayer::calc_derivative(map<string, Tensor*>& p_derivative)
{
	Tensor dy = _y->derivative();
	p_derivative[_id] = NeuronOperator::init_auxiliary_parameter(p_derivative[_id], _batch_size, _dim);
	p_derivative[_id]->override(&dy);
}

void CoreLayer::override(BaseLayer * p_source)
{
	CoreLayer* source = dynamic_cast<CoreLayer*>(p_source);

}

void CoreLayer::init(vector<BaseLayer*>& p_input_layers)
{
	BaseLayer::init(p_input_layers);

	_W = new Param(IDGen::instance().next(), new Tensor({_dim, _in_dim}, Tensor::ZERO));
	_initializer->init(_W->get_data());
	add_param(_W->get_id(), _W->get_data());
}

json CoreLayer::get_json() const
{
	json data = BaseLayer::get_json();

	//data["group"] = _group->get_json();

	return data;
}

CoreLayer::CoreLayer(CoreLayer* p_source) : BaseLayer(p_source)
{
	_type = CORE;
}

CoreLayer* CoreLayer::clone()
{
	return new CoreLayer(this);
}