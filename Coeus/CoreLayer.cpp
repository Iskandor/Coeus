#include "CoreLayer.h"
#include "IDGen.h"
#include "TensorOperator.h"
#include "ActivationFunctionFactory.h"
#include "TensorInitializer.h"
#include "IOUtils.h"

using namespace Coeus;

CoreLayer::CoreLayer(const string& p_id, const int p_dim, const ACTIVATION p_activation, TensorInitializer* p_initializer, const int p_in_dim) : BaseLayer(p_id, p_dim, { p_in_dim })
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
	_y = new NeuronOperator(p_data["y"]);
	add_param(_y);
	_W = IOUtils::load_param(p_data["W"]);
	_initializer = nullptr;
}

CoreLayer::CoreLayer(CoreLayer &p_copy) : BaseLayer(p_copy._id, p_copy._dim, { p_copy._in_dim }) {
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

void CoreLayer::calc_gradient(map<string, Tensor>& p_gradient_map, map<string, Tensor*>& p_delta_map, map<string, Tensor*>& p_derivative_map)
{
	Tensor*	 delta_out = _y->get_function()->backward(p_delta_map[_id]);

	if (_batch)
	{
		TensorOperator::instance().m_reduce(p_gradient_map[_y->get_bias()->get_id()].arr(), delta_out->arr(), _batch_size, _dim);
		TensorOperator::instance().full_gradient_b(_batch_size, _input->arr(), delta_out->arr(), p_gradient_map[_W->get_id()].arr(), _dim, _in_dim);
	}
	else
	{
		p_gradient_map[_y->get_bias()->get_id()].override(delta_out);
		TensorOperator::instance().full_gradient_s(_input->arr(), delta_out->arr(), p_gradient_map[_W->get_id()].arr(), _dim, _in_dim);
	}

	Tensor*	 delta_in = nullptr;

	if (!_input_layer.empty())
	{
		delta_in = NeuronOperator::init_auxiliary_parameter(delta_in, _batch_size, _in_dim);

		if (_batch)
		{
			TensorOperator::instance().full_delta_b(_batch_size, delta_in->arr(), delta_out->arr(), _W->get_data()->arr(), _dim, _in_dim);
		}
		else
		{
			TensorOperator::instance().full_delta_s(delta_in->arr(), delta_out->arr(), _W->get_data()->arr(), _dim, _in_dim);
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

	delete delta_out;
}

void CoreLayer::calc_derivative(map<string, Tensor*>& p_derivative)
{
}

void CoreLayer::override(BaseLayer * p_source)
{
	CoreLayer* source = dynamic_cast<CoreLayer*>(p_source);

}

void CoreLayer::init(vector<BaseLayer*>& p_input_layers)
{
	BaseLayer::init(p_input_layers);

	if (_W == nullptr)
	{
		_W = new Param(IDGen::instance().next(), new Tensor({ _dim, _in_dim }, Tensor::ZERO));
		_initializer->init(_W->get_data());
	}
	add_param(_W->get_id(), _W->get_data());

	cout << _id << " " << _in_dim << " - " << _dim << endl;
}

json CoreLayer::get_json() const
{
	json data = BaseLayer::get_json();

	data["W"] = IOUtils::save_param(_W);
	data["y"] = _y->get_json();

	return data;
}

Tensor* CoreLayer::get_dim_tensor()
{
	if (_dim_tensor == nullptr)
	{
		_dim_tensor = new Tensor({ 1 }, Tensor::VALUE, _dim);
	}

	return _dim_tensor;
}

CoreLayer::CoreLayer(CoreLayer* p_source) : BaseLayer(p_source)
{
	_type = CORE;
}

CoreLayer* CoreLayer::clone()
{
	return new CoreLayer(this);
}