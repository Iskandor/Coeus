#include "RecurrentLayer.h"
#include "IDGen.h"
#include "NeuronOperator.h"
#include "TensorOperator.h"
#include "IOUtils.h"

using namespace Coeus;

RecurrentLayer::RecurrentLayer(const string& p_id, const int p_dim, const ACTIVATION p_activation, TensorInitializer* p_initializer, const int p_in_dim) : BaseLayer(p_id, p_dim, { p_in_dim })
{
	_type = RECURRENT;
	_is_recurrent = true;

	_initializer = p_initializer;

	_y = new NeuronOperator(p_dim, p_activation);
	add_param(_y);	
	_W = nullptr;
	_context = nullptr;	
}

RecurrentLayer::RecurrentLayer(RecurrentLayer& p_copy, const bool p_clone) : BaseLayer(p_copy._id, p_copy._dim, { p_copy._in_dim }) {
	_type = RECURRENT;
	_is_recurrent = true;
	_y = new NeuronOperator(*p_copy._y, p_clone);
	add_param(_y);

	if (p_clone)
	{
		_W = new Param(IDGen::instance().next(), new Tensor(*p_copy._W->get_data()));
	}
	else
	{
		_W = new Param(p_copy._W->get_id(), p_copy._W->get_data());
	}
	add_param(_W);
	
	_initializer = new TensorInitializer(*p_copy._initializer);
	_context = nullptr;
}

RecurrentLayer::RecurrentLayer(const json& p_data) : BaseLayer(p_data)
{
	_type = RECURRENT;
	_is_recurrent = true;
	_y = new NeuronOperator(p_data["y"]);
	add_param(_y);
	_W = IOUtils::load_param(p_data["W"]);
	add_param(_W);
	_initializer = nullptr;
	_context = nullptr;
}

RecurrentLayer::~RecurrentLayer()
{
	delete _y;
	delete _W;
	delete _context;
	delete _initializer;
}

RecurrentLayer* RecurrentLayer::copy(const bool p_clone)
{
	return new RecurrentLayer(*this, p_clone);
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

	if (_mode == BPTT)
	{
		map<string, Tensor*> values;
		values["input"] = new Tensor(*_input);
		values[_y->get_id()] = new Tensor(*_y->get_output());
		_bptt_values.push(values);
	}
}

void RecurrentLayer::calc_gradient(Gradient& p_gradient_map, map<string, Tensor*>& p_derivative_map)
{
	BaseLayer::calc_gradient(p_gradient_map, p_derivative_map);

	Tensor*	df = nullptr;

	if (_mode == NONE)
	{
		df = _y->get_function()->backward(_delta_out);
		TensorOperator::instance().full_w_gradient(_batch_size, _input->arr(), df->arr(), p_gradient_map[_W->get_id()].arr(), _dim, _in_dim, false);
		TensorOperator::instance().full_b_gradient(_batch_size, df->arr(), p_gradient_map[_y->get_bias()->get_id()].arr(), _dim, false);
	}

	if (_mode == BPTT)
	{
		map<string, Tensor*> values = _bptt_values.top();

		df = _y->get_function()->backward(_delta_out, values[_y->get_id()]);
		TensorOperator::instance().full_w_gradient(_batch_size, values["input"]->arr(), df->arr(), p_gradient_map[_W->get_id()].arr(), _dim, _in_dim, true);
		TensorOperator::instance().full_b_gradient(_batch_size, df->arr(), p_gradient_map[_y->get_bias()->get_id()].arr(), _dim, true);

		_bptt_values.pop();

		for (auto it : values)
		{
			delete it.second;
		}
	}

	Tensor*	 delta_in = nullptr;
	
	if (!_input_layer.empty())
	{
		delta_in = NeuronOperator::init_auxiliary_parameter(delta_in, _batch_size, _in_dim);

		TensorOperator::instance().full_delta(_batch_size, delta_in->arr(), df->arr(), _W->get_data()->arr(), _dim, _in_dim);

		int index = _input_dim;

		for (auto it : _input_layer)
		{
			_delta_in[it->get_id()] = NeuronOperator::init_auxiliary_parameter(_delta_in[it->get_id()], _batch_size, it->get_dim());
			delta_in->splice(index, _delta_in[it->get_id()]);
			index += it->get_dim();
		}

		delete delta_in;
	}
}

void RecurrentLayer::calc_derivative(map<string, Tensor*>& p_derivative)
{
}

void RecurrentLayer::reset()
{
	if (_context != nullptr) _context->fill(0);
}

void RecurrentLayer::init(vector<BaseLayer*>& p_input_layers, vector<BaseLayer*>& p_output_layers)
{
	BaseLayer::init(p_input_layers, p_output_layers);

	_input_layer.push_back(this);
	_in_dim += _dim;

	if (_W == nullptr) {
		_W = new Param(IDGen::instance().next(), new Tensor({ _dim, _in_dim }, Tensor::ZERO));
		_initializer->init(_W->get_data());
		add_param(_W->get_id(), _W->get_data());
	}

	_output_layer.push_back(this);
}

json RecurrentLayer::get_json() const
{
	json data = BaseLayer::get_json();

	data["W"] = IOUtils::save_param(_W);
	data["y"] = _y->get_json();

	return data;
}

Tensor* RecurrentLayer::get_dim_tensor()
{
	if (_dim_tensor == nullptr)
	{
		_dim_tensor = new Tensor({ 1 }, Tensor::VALUE, _dim);
	}

	return _dim_tensor;
}
