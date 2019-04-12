#include "RecurrentLayer.h"
#include "IDGen.h"
#include "NeuronOperator.h"
#include "TensorOperator.h"

using namespace Coeus;

RecurrentLayer::RecurrentLayer(const string& p_id, const int p_dim, const ACTIVATION p_activation, TensorInitializer* p_initializer, const int p_in_dim) : BaseLayer(p_id, p_dim, p_in_dim)
{
	_type = RECURRENT;

	_initializer = p_initializer;

	_y = new NeuronOperator(p_dim, p_activation);
	add_param(_y);	

	_context = nullptr;	
}

RecurrentLayer::RecurrentLayer(RecurrentLayer& p_copy) : BaseLayer(p_copy._id, p_copy._dim, p_copy._in_dim) {
	_type = RECURRENT;
	_y = new NeuronOperator(*p_copy._y);
	_W = new Param(*p_copy._W);
	_initializer = p_copy._initializer;
	_context = nullptr;
}

RecurrentLayer::RecurrentLayer(const json& p_data) : BaseLayer(p_data)
{
	_type = RECURRENT;
}

RecurrentLayer::~RecurrentLayer()
{
	delete _y;
	delete _W;
	delete _context;
	delete _initializer;
}

RecurrentLayer* RecurrentLayer::clone()
{
	return new RecurrentLayer(this);
}

void RecurrentLayer::activate()
{
	_context = NeuronOperator::init_auxiliary_parameter(_context, _batch_size, _dim);

	_input->push_back(_context);
	_input->reset_index();

	_y->integrate(_input, _W->get_data());
	_y->activate();

	_output = _y->get_output();
	_context->override(_y->get_output());
}

void RecurrentLayer::calc_delta(map<string, Tensor*>& p_delta_map, map<string, Tensor*>& p_derivative_map)
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
			TensorOperator::instance().full_delta_b(_batch_size, delta_in->arr(), delta_out->arr(), _W->get_data()->arr(), _in_derivative->arr(), _dim, _in_dim);
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

void RecurrentLayer::calc_gradient(map<string, Tensor>& p_gradient_map, map<string, Tensor*>& p_delta_map, map<string, Tensor*>& p_derivative_map)
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

void RecurrentLayer::calc_derivative(map<string, Tensor*>& p_derivative)
{
	Tensor dy = _y->derivative();
	p_derivative[_id] = NeuronOperator::init_auxiliary_parameter(p_derivative[_id], _batch_size, _dim);
	p_derivative[_id]->override(&dy);
}


void RecurrentLayer::override(BaseLayer * p_source)
{
	const RecurrentLayer *source = dynamic_cast<RecurrentLayer*>(p_source);
	_y->get_bias()->get_data()->override(source->_y->get_bias()->get_data());
	_W->get_data()->override(source->_W->get_data());
}

void RecurrentLayer::reset()
{
	if (_context != nullptr) _context->fill(0);
}

void RecurrentLayer::init(vector<BaseLayer*>& p_input_layers)
{
	BaseLayer::init(p_input_layers);

	_in_dim += _dim;

	_W = new Param(IDGen::instance().next(), new Tensor({ _dim, _in_dim }, Tensor::ZERO));
	_initializer->init(_W->get_data());
	add_param(_W->get_id(), _W->get_data());
}

json RecurrentLayer::get_json() const
{
	json data = BaseLayer::get_json();

	//data["group"] = _group->get_json();

	return data;
}

RecurrentLayer::RecurrentLayer(RecurrentLayer* p_source) : BaseLayer(p_source)
{
	_type = RECURRENT;
}
