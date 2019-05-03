#include "LSTMLayer.h"
#include "IDGen.h"
#include "ActivationFunctionFactory.h"
#include "TensorInitializer.h"
#include "TensorOperator.h"
#include "NeuronOperator.h"
#include "IOUtils.h"

using namespace Coeus;


LSTMLayer::LSTMLayer(const string& p_id, const int p_dim, const ACTIVATION p_activation, TensorInitializer* p_initializer, const int p_in_dim) : BaseLayer(p_id, p_dim, p_in_dim)
{
	_type = LSTM;

	_activation_function = ActivationFunctionFactory::create_function(p_activation);
	_initializer = p_initializer;

	_cec = new NeuronOperator(p_dim, TANH);
	add_param(_cec);
	_ig = new NeuronOperator(p_dim, SIGMOID);
	add_param(_ig);
	_fg = new NeuronOperator(p_dim, SIGMOID);
	add_param(_fg);
	_og = new NeuronOperator(p_dim, SIGMOID);
	add_param(_og);

	_Wxc = nullptr;
	_Wxig = nullptr;
	_Wxfg = nullptr;
	_Wxog = nullptr;
	
	_state = nullptr;
	_state_error = nullptr;
	_context = nullptr;
}

LSTMLayer::LSTMLayer(json p_data) : BaseLayer(p_data)
{
	_type = LSTM;

	_activation_function = IOUtils::init_activation_function(p_data["f"]);
	_initializer = nullptr;

	_cec = new NeuronOperator(p_data["cec"]);
	add_param(_cec);
	_ig = new NeuronOperator(p_data["ig"]);
	add_param(_ig);
	_fg = new NeuronOperator(p_data["fg"]);
	add_param(_fg);
	_og = new NeuronOperator(p_data["og"]);
	add_param(_og);

	_Wxc = IOUtils::load_param(p_data["Wxc"]);
	_Wxig = IOUtils::load_param(p_data["Wxig"]);
	_Wxfg = IOUtils::load_param(p_data["Wxfg"]);
	_Wxog = IOUtils::load_param(p_data["Wxog"]);

	_state = nullptr;
	_state_error = nullptr;
	_context = nullptr;
}

LSTMLayer::~LSTMLayer()
{
	delete _cec;
	delete _ig;
	delete _fg;
	delete _og;

	delete _Wxc;
	delete _Wxig;
	delete _Wxfg;
	delete _Wxog;

	delete _state;
	delete _state_error;
	delete _context;
}

LSTMLayer* LSTMLayer::clone()
{
	return new LSTMLayer(this);
}

void LSTMLayer::init(vector<BaseLayer*>& p_input_layers)
{
	BaseLayer::init(p_input_layers);

	_in_dim += _dim;

	if (_Wxc == nullptr) {
		_Wxc = new Param(IDGen::instance().next(), new Tensor({ _dim, _in_dim }, Tensor::ZERO));
		_initializer->init(_Wxc->get_data());
	}
	add_param(_Wxc->get_id(), _Wxc->get_data());

	if (_Wxig == nullptr) {
		_Wxig = new Param(IDGen::instance().next(), new Tensor({ _dim, _in_dim }, Tensor::ZERO));
		_initializer->init(_Wxig->get_data());
	}
	add_param(_Wxig->get_id(), _Wxig->get_data());

	if (_Wxfg == nullptr) {
		_Wxfg = new Param(IDGen::instance().next(), new Tensor({ _dim, _in_dim }, Tensor::ZERO));
		_initializer->init(_Wxfg->get_data());
	}
	add_param(_Wxfg->get_id(), _Wxfg->get_data());

	if (_Wxog == nullptr) {
		_Wxog = new Param(IDGen::instance().next(), new Tensor({ _dim, _in_dim }, Tensor::ZERO));
		_initializer->init(_Wxog->get_data());
	}
	add_param(_Wxog->get_id(), _Wxog->get_data());
}

void LSTMLayer::activate()
{	
	_context = NeuronOperator::init_auxiliary_parameter(_context, _batch_size, _dim);
	_state = NeuronOperator::init_auxiliary_parameter(_state, _batch_size, _dim);
	_state_error = NeuronOperator::init_auxiliary_parameter(_state_error, _batch_size, _dim);
	_output = NeuronOperator::init_auxiliary_parameter(_output, _batch_size, _dim);

	_input->push_back(_context);
	_input->reset_index();

	_ig->integrate(_input, _Wxig->get_data());
	_ig->activate();
	
	_fg->integrate(_input, _Wxfg->get_data());
	_fg->activate();
	
	_og->integrate(_input, _Wxog->get_data());
	_og->activate();

	_cec->integrate(_input, _Wxc->get_data());
	_cec->activate();

	if (_batch_size == 1)
	{
		TensorOperator::instance().lstm_state_s(_state->arr(), _ig->get_output()->arr(), _fg->get_output()->arr(), _cec->get_output()->arr(), _dim);
		Tensor *ac = _activation_function->forward(_state);
		TensorOperator::instance().vv_ewprod(_og->get_output()->arr(), ac->arr(), _output->arr(), _dim);
	}
	if (_batch_size > 1)
	{
		TensorOperator::instance().lstm_state_b(_batch_size, _state->arr(), _ig->get_output()->arr(), _fg->get_output()->arr(), _cec->get_output()->arr(), _dim);
		Tensor *ac = _activation_function->forward(_state);
		TensorOperator::instance().vv_ewprod(_og->get_output()->arr(), ac->arr(), _output->arr(), _batch_size * _dim);
	}

	_context->override(_output);
}

void LSTMLayer::calc_delta(map<string, Tensor*>& p_delta_map, map<string, Tensor*>& p_derivative_map)
{
	_in_derivative->reset_index();
	for (auto it : _input_layer)
	{
		_in_derivative->push_back(p_derivative_map[it->get_id()]);
	}

	Tensor* h = _activation_function->forward(_state);
	Tensor dh = _activation_function->derivative(*_state);

	Tensor*	 delta_out = p_delta_map[_id];
	Tensor*	 delta_in = nullptr;
	Tensor*	 delta = nullptr;

	p_delta_map[_og->get_id()] = NeuronOperator::init_auxiliary_parameter(p_delta_map[_og->get_id()], _batch_size, _dim);
	p_delta_map[_cec->get_id()] = NeuronOperator::init_auxiliary_parameter(p_delta_map[_cec->get_id()], _batch_size, _dim);
	p_delta_map[_ig->get_id()] = NeuronOperator::init_auxiliary_parameter(p_delta_map[_ig->get_id()], _batch_size, _dim);
	p_delta_map[_fg->get_id()] = NeuronOperator::init_auxiliary_parameter(p_delta_map[_fg->get_id()], _batch_size, _dim);

	_state_error = NeuronOperator::init_auxiliary_parameter(_state_error, _batch_size, _dim);

	if (_batch)
	{
		TensorOperator::instance().lstm_delta_b(_batch_size, p_delta_map[_og->get_id()]->arr(), _og->derivative().arr(), h->arr(), delta_out->arr(), _dim);
		TensorOperator::instance().lstm_delta_b(_batch_size, _state_error->arr(), _og->get_output()->arr(), dh.arr(), delta_out->arr(), _dim);
		p_delta_map[_cec->get_id()]->override(p_derivative_map[_cec->get_bias()->get_id()]);

		TensorOperator::instance().vv_ewprod(_state_error->arr(), p_derivative_map[_ig->get_bias()->get_id()]->arr(), p_delta_map[_ig->get_id()]->arr(), _batch_size * _dim);
		TensorOperator::instance().vv_ewprod(_state_error->arr(), p_derivative_map[_fg->get_bias()->get_id()]->arr(), p_delta_map[_fg->get_id()]->arr(), _batch_size * _dim);

		if (!_input_layer.empty())
		{
			delta_in = NeuronOperator::init_auxiliary_parameter(delta_in, _batch_size, _in_dim);
			delta = NeuronOperator::init_auxiliary_parameter(delta, _batch_size, _in_dim);

			TensorOperator::instance().full_delta_b(_batch_size, delta->arr(), p_delta_map[_cec->get_id()]->arr(), _Wxc->get_data()->arr(), _in_derivative->arr(), _dim, _in_dim);
			TensorOperator::instance().vv_add(delta->arr(), delta_in->arr(), delta_in->arr(), _batch_size *_in_dim);
			TensorOperator::instance().full_delta_b(_batch_size, delta->arr(), p_delta_map[_ig->get_id()]->arr(), _Wxig->get_data()->arr(), _in_derivative->arr(), _dim, _in_dim);
			TensorOperator::instance().vv_add(delta->arr(), delta_in->arr(), delta_in->arr(), _batch_size *_in_dim);
			TensorOperator::instance().full_delta_b(_batch_size, delta->arr(), p_delta_map[_fg->get_id()]->arr(), _Wxfg->get_data()->arr(), _in_derivative->arr(), _dim, _in_dim);
			TensorOperator::instance().vv_add(delta->arr(), delta_in->arr(), delta_in->arr(), _batch_size *_in_dim);
			TensorOperator::instance().full_delta_b(_batch_size, delta->arr(), p_delta_map[_og->get_id()]->arr(), _Wxog->get_data()->arr(), _in_derivative->arr(), _dim, _in_dim);
			TensorOperator::instance().vv_add(delta->arr(), delta_in->arr(), delta_in->arr(), _batch_size *_in_dim);

			int index = 0;

			for (auto it : _input_layer)
			{
				p_delta_map[it->get_id()] = NeuronOperator::init_auxiliary_parameter(p_delta_map[it->get_id()], _batch_size, it->get_dim());
				delta_in->splice(index, p_delta_map[it->get_id()]);
				index += it->get_dim();
			}

			delete delta_in;
			delete delta;
		}
	}
	else
	{
		TensorOperator::instance().lstm_delta_s(p_delta_map[_og->get_id()]->arr(), _og->derivative().arr(), h->arr(), delta_out->arr(), _dim);
		TensorOperator::instance().lstm_delta_s(_state_error->arr(), _og->get_output()->arr(), dh.arr(), delta_out->arr(), _dim);

		p_delta_map[_cec->get_id()]->override(p_derivative_map[_cec->get_bias()->get_id()]);
		TensorOperator::instance().vv_ewprod(_state_error->arr(), p_derivative_map[_ig->get_bias()->get_id()]->arr(), p_delta_map[_ig->get_id()]->arr(), _dim);
		TensorOperator::instance().vv_ewprod(_state_error->arr(), p_derivative_map[_fg->get_bias()->get_id()]->arr(), p_delta_map[_fg->get_id()]->arr(), _dim);

		if (!_input_layer.empty())
		{
			delta_in = NeuronOperator::init_auxiliary_parameter(delta_in, _batch_size, _in_dim);
			delta = NeuronOperator::init_auxiliary_parameter(delta, _batch_size, _in_dim);

			TensorOperator::instance().full_delta_s(delta->arr(), p_delta_map[_cec->get_id()]->arr(), _Wxc->get_data()->arr(), _in_derivative->arr(), _dim, _in_dim);
			TensorOperator::instance().vv_add(delta->arr(), delta_in->arr(), delta_in->arr(), _in_dim);
			TensorOperator::instance().full_delta_s(delta->arr(), p_delta_map[_ig->get_id()]->arr(), _Wxig->get_data()->arr(), _in_derivative->arr(), _dim, _in_dim);
			TensorOperator::instance().vv_add(delta->arr(), delta_in->arr(), delta_in->arr(), _in_dim);
			TensorOperator::instance().full_delta_s(delta->arr(), p_delta_map[_fg->get_id()]->arr(), _Wxfg->get_data()->arr(), _in_derivative->arr(), _dim, _in_dim);
			TensorOperator::instance().vv_add(delta->arr(), delta_in->arr(), delta_in->arr(), _in_dim);
			TensorOperator::instance().full_delta_s(delta->arr(), p_delta_map[_og->get_id()]->arr(), _Wxog->get_data()->arr(), _in_derivative->arr(), _dim, _in_dim);
			TensorOperator::instance().vv_add(delta->arr(), delta_in->arr(), delta_in->arr(), _in_dim);

			int index = 0;

			for (auto it : _input_layer)
			{
				p_delta_map[it->get_id()] = NeuronOperator::init_auxiliary_parameter(p_delta_map[it->get_id()], _batch_size, it->get_dim());
				delta_in->splice(index, p_delta_map[it->get_id()]);
				index += it->get_dim();
			}

			delete delta_in;
			delete delta;
		}
	}
}

void LSTMLayer::calc_gradient(map<string, Tensor>& p_gradient_map, map<string, Tensor*>& p_delta_map, map<string, Tensor*>& p_derivative_map)
{
	Tensor* gWxig = &p_gradient_map[_Wxig->get_id()];
	Tensor* gWxfg = &p_gradient_map[_Wxfg->get_id()];
	Tensor* gWxog = &p_gradient_map[_Wxog->get_id()];
	Tensor* gWxc = &p_gradient_map[_Wxc->get_id()];

	Tensor* dWxig = p_derivative_map[_Wxig->get_id()];
	Tensor* dWxfg = p_derivative_map[_Wxfg->get_id()];
	Tensor* dWxc = p_derivative_map[_Wxc->get_id()];

	if (_batch)
	{
		TensorOperator::instance().full_gradient_b(_batch_size, _input->arr(), p_delta_map[_og->get_id()]->arr(), gWxog->arr(), _dim, _in_dim);
		TensorOperator::instance().lstm_gradient_b(_batch_size, gWxig->arr(), _state_error->arr(), dWxig->arr(), _dim, _in_dim);
		TensorOperator::instance().lstm_gradient_b(_batch_size, gWxfg->arr(), _state_error->arr(), dWxfg->arr(), _dim, _in_dim);
		TensorOperator::instance().lstm_gradient_b(_batch_size, gWxc->arr(), _state_error->arr(), dWxc->arr(), _dim, _in_dim);

		TensorOperator::instance().m_reduce(p_gradient_map[_og->get_bias()->get_id()].arr(), p_delta_map[_og->get_id()]->arr(), _batch_size, _dim);
		TensorOperator::instance().m_reduce(p_gradient_map[_cec->get_bias()->get_id()].arr(), p_delta_map[_cec->get_id()]->arr(), _batch_size, _dim);
		TensorOperator::instance().m_reduce(p_gradient_map[_ig->get_bias()->get_id()].arr(), p_delta_map[_ig->get_id()]->arr(), _batch_size, _dim);
		TensorOperator::instance().m_reduce(p_gradient_map[_fg->get_bias()->get_id()].arr(), p_delta_map[_fg->get_id()]->arr(), _batch_size, _dim);
	}
	else
	{
		TensorOperator::instance().full_gradient_s(_input->arr(), p_delta_map[_og->get_id()]->arr(), gWxog->arr(), _dim, _in_dim);
		TensorOperator::instance().lstm_gradient_s(gWxig->arr(), _state_error->arr(), dWxig->arr(), _dim, _in_dim);
		TensorOperator::instance().lstm_gradient_s(gWxfg->arr(), _state_error->arr(), dWxfg->arr(), _dim, _in_dim);
		TensorOperator::instance().lstm_gradient_s(gWxc->arr(), _state_error->arr(), dWxc->arr(), _dim, _in_dim);

		p_gradient_map[_og->get_bias()->get_id()].override(p_delta_map[_og->get_id()]);
		p_gradient_map[_cec->get_bias()->get_id()].override(p_delta_map[_cec->get_id()]);
		p_gradient_map[_ig->get_bias()->get_id()].override(p_delta_map[_ig->get_id()]);
		p_gradient_map[_fg->get_bias()->get_id()].override(p_delta_map[_fg->get_id()]);
	}

}

void LSTMLayer::calc_derivative(map<string, Tensor*>& p_derivative)
{
	Tensor dy = _activation_function->derivative(*_output);
	p_derivative[_id] = NeuronOperator::init_auxiliary_parameter(p_derivative[_id], _batch_size, _dim);
	p_derivative[_id]->override(&dy);

	Tensor dcec = _cec->derivative();
	p_derivative[_cec->get_id()] = NeuronOperator::init_auxiliary_parameter(p_derivative[_cec->get_id()], _batch_size, _dim);
	p_derivative[_cec->get_id()]->override(&dcec);

	Tensor dig = _ig->derivative();
	p_derivative[_ig->get_id()] = NeuronOperator::init_auxiliary_parameter(p_derivative[_ig->get_id()], _batch_size, _dim);
	p_derivative[_ig->get_id()]->override(&dig);

	Tensor dfg = _fg->derivative();
	p_derivative[_fg->get_id()] = NeuronOperator::init_auxiliary_parameter(p_derivative[_fg->get_id()], _batch_size, _dim);
	p_derivative[_fg->get_id()]->override(&dfg);

	Tensor dog = _og->derivative();
	p_derivative[_og->get_id()] = NeuronOperator::init_auxiliary_parameter(p_derivative[_og->get_id()], _batch_size, _dim);
	p_derivative[_og->get_id()]->override(&dog);

	p_derivative[_Wxc->get_id()] = NeuronOperator::init_auxiliary_parameter(p_derivative[_Wxc->get_id()], _batch_size, _dim * _in_dim);
	p_derivative[_Wxfg->get_id()] = NeuronOperator::init_auxiliary_parameter(p_derivative[_Wxfg->get_id()], _batch_size, _dim * _in_dim);
	p_derivative[_Wxig->get_id()] = NeuronOperator::init_auxiliary_parameter(p_derivative[_Wxig->get_id()], _batch_size, _dim * _in_dim);
	p_derivative[_cec->get_bias()->get_id()] = NeuronOperator::init_auxiliary_parameter(p_derivative[_cec->get_bias()->get_id()], _batch_size, _dim);
	p_derivative[_fg->get_bias()->get_id()] = NeuronOperator::init_auxiliary_parameter(p_derivative[_fg->get_bias()->get_id()], _batch_size, _dim);
	p_derivative[_ig->get_bias()->get_id()] = NeuronOperator::init_auxiliary_parameter(p_derivative[_ig->get_bias()->get_id()], _batch_size, _dim);

	if (_batch)
	{
		Tensor bias_input = Tensor::Ones({ _batch_size, 1 });

		TensorOperator::instance().lstm_derivative_b(_batch_size, p_derivative[_Wxc->get_id()]->arr(), _fg->get_output()->arr(), dcec.arr(), _ig->get_output()->arr(), _input->arr(), _dim, _in_dim);
		TensorOperator::instance().lstm_derivative_b(_batch_size, p_derivative[_Wxig->get_id()]->arr(), _fg->get_output()->arr(), _cec->get_output()->arr(), dig.arr(), _input->arr(), _dim, _in_dim);
		TensorOperator::instance().lstm_derivative_b(_batch_size, p_derivative[_Wxfg->get_id()]->arr(), _fg->get_output()->arr(), _state->arr(), dfg.arr(), _input->arr(), _dim, _in_dim);

		TensorOperator::instance().lstm_derivative_b(_batch_size, p_derivative[_cec->get_bias()->get_id()]->arr(), _fg->get_output()->arr(), dcec.arr(), _ig->get_output()->arr(), bias_input.arr(), _dim, 1);
		TensorOperator::instance().lstm_derivative_b(_batch_size, p_derivative[_ig->get_bias()->get_id()]->arr(), _fg->get_output()->arr(), _cec->get_output()->arr(), dig.arr(), bias_input.arr(), _dim, 1);
		TensorOperator::instance().lstm_derivative_b(_batch_size, p_derivative[_fg->get_bias()->get_id()]->arr(), _fg->get_output()->arr(), _state->arr(), dfg.arr(), bias_input.arr(), _dim, 1);
	}
	else
	{
		Tensor bias_input = Tensor::Ones({ 1 });

		TensorOperator::instance().lstm_derivative_s(p_derivative[_Wxc->get_id()]->arr(), _fg->get_output()->arr(), dcec.arr(), _ig->get_output()->arr(), _input->arr(), _dim, _in_dim);
		TensorOperator::instance().lstm_derivative_s(p_derivative[_Wxig->get_id()]->arr(), _fg->get_output()->arr(), _cec->get_output()->arr(), dig.arr(), _input->arr(), _dim, _in_dim);
		TensorOperator::instance().lstm_derivative_s(p_derivative[_Wxfg->get_id()]->arr(), _fg->get_output()->arr(), _state->arr(), dfg.arr(), _input->arr(), _dim, _in_dim);

		TensorOperator::instance().lstm_derivative_s(p_derivative[_cec->get_bias()->get_id()]->arr(), _fg->get_output()->arr(), dcec.arr(), _ig->get_output()->arr(), bias_input.arr(), _dim, 1);
		TensorOperator::instance().lstm_derivative_s(p_derivative[_ig->get_bias()->get_id()]->arr(), _fg->get_output()->arr(), _cec->get_output()->arr(), dig.arr(), bias_input.arr(), _dim, 1);
		TensorOperator::instance().lstm_derivative_s(p_derivative[_fg->get_bias()->get_id()]->arr(), _fg->get_output()->arr(), _state->arr(), dfg.arr(), bias_input.arr(), _dim, 1);
	}
}

void LSTMLayer::override(BaseLayer* p_source)
{
//#TODO doplnit prepis parametrov LSTM siete
}

void LSTMLayer::reset()
{
	if (_context != nullptr) _context->fill(0);
	if (_state != nullptr) _state->fill(0);
}

json LSTMLayer::get_json() const
{
	json data = BaseLayer::get_json();

	data["f"] = _activation_function->get_json();

	data["cec"] = _cec->get_json();
	data["ig"] = _ig->get_json();
	data["fg"] = _fg->get_json();
	data["og"] = _og->get_json();

	data["Wxc"] = IOUtils::save_param(_Wxc);
	data["Wxig"] = IOUtils::save_param(_Wxig);
	data["Wxfg"] = IOUtils::save_param(_Wxfg);
	data["Wxog"] = IOUtils::save_param(_Wxog);

	return data;
}

LSTMLayer::LSTMLayer(LSTMLayer* p_source) : BaseLayer(p_source)
{
	_type = LSTM;

	/*
	_input_gate = add_group<SimpleCellGroup>(new SimpleCellGroup(p_source->_input_gate));
	_output_gate = add_group<SimpleCellGroup>(new SimpleCellGroup(p_source->_output_gate));
	_forget_gate = add_group<SimpleCellGroup>(new SimpleCellGroup(p_source->_forget_gate));
	_cec = add_group<LSTMCellGroup>(new LSTMCellGroup(p_source->_cec, _input_gate, _output_gate, _forget_gate));
	_context = add_group<SimpleCellGroup>(new SimpleCellGroup(p_source->_context));

	_aux_input = add_group<SimpleCellGroup>(new SimpleCellGroup(p_source->_aux_input));
	_in_input_gate = add_connection(p_source->_in_input_gate->clone());
	_in_output_gate = add_connection(p_source->_in_output_gate->clone());
	_in_forget_gate = add_connection(p_source->_in_forget_gate->clone());

	_output_group = _cec;

	_ct_cec = add_connection(p_source->_ct_cec->clone());
	*/
}
