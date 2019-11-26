#include "LSTMLayer.h"
#include "IDGen.h"
#include "ActivationFunctionFactory.h"
#include "TensorInitializer.h"
#include "TensorOperator.h"
#include "NeuronOperator.h"
#include "IOUtils.h"

using namespace Coeus;


LSTMLayer::LSTMLayer(const string& p_id, const int p_dim, const ACTIVATION p_activation, TensorInitializer* p_initializer, const int p_in_dim) : BaseLayer(p_id, p_dim, { p_in_dim })
{
	_type = LSTM;
	_is_recurrent = true;

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
	_context = nullptr;
}

LSTMLayer::LSTMLayer(json p_data) : BaseLayer(p_data)
{
	_type = LSTM;
	_is_recurrent = true;

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
	add_param(_Wxc);
	add_param(_Wxig);
	add_param(_Wxfg);
	add_param(_Wxog);

	_state = nullptr;
	_context = nullptr;
}

LSTMLayer::LSTMLayer(LSTMLayer& p_copy, const bool p_clone) : BaseLayer(p_copy._id, p_copy._dim, { p_copy._in_dim })
{
	_type = LSTM;
	_is_recurrent = true;

	_activation_function = ActivationFunctionFactory::create_function(p_copy._activation_function->get_type());
	_initializer = new TensorInitializer(*p_copy._initializer);

	_cec = new NeuronOperator(*p_copy._cec, p_clone);
	add_param(_cec);
	_ig = new NeuronOperator(*p_copy._ig, p_clone);
	add_param(_ig);
	_fg = new NeuronOperator(*p_copy._fg, p_clone);
	add_param(_fg);
	_og = new NeuronOperator(*p_copy._og, p_clone);
	add_param(_og);

	if (p_clone)
	{
		_Wxc = new Param(IDGen::instance().next(), new Tensor(*p_copy._Wxc->get_data()));
		_Wxig = new Param(IDGen::instance().next(), new Tensor(*p_copy._Wxig->get_data()));
		_Wxfg = new Param(IDGen::instance().next(), new Tensor(*p_copy._Wxfg->get_data()));
		_Wxog = new Param(IDGen::instance().next(), new Tensor(*p_copy._Wxog->get_data()));
	}
	else
	{
		_Wxc = new Param(p_copy._Wxc->get_id(), p_copy._Wxc->get_data());
		_Wxig = new Param(p_copy._Wxig->get_id(), p_copy._Wxig->get_data());
		_Wxfg = new Param(p_copy._Wxfg->get_id(), p_copy._Wxfg->get_data());
		_Wxog = new Param(p_copy._Wxog->get_id(), p_copy._Wxog->get_data());
	}
	add_param(_Wxc);
	add_param(_Wxig);
	add_param(_Wxfg);
	add_param(_Wxog);
	

	_state = nullptr;
	_context = nullptr;
}

LSTMLayer* LSTMLayer::copy(const bool p_clone)
{
	return new LSTMLayer(*this, p_clone);
}

LSTMLayer::~LSTMLayer()
{
	delete _cec;
	delete _ig;
	delete _fg;
	delete _og;
	
	delete _state;
	delete _context;
}

void LSTMLayer::init(vector<BaseLayer*>& p_input_layers, vector<BaseLayer*>& p_output_layers)
{
	BaseLayer::init(p_input_layers, p_output_layers);

	_in_dim += _dim;

	if (_Wxc == nullptr) {
		_Wxc = new Param(IDGen::instance().next(), new Tensor({ _dim, _in_dim }, Tensor::ZERO));
		_initializer->init(_Wxc->get_data());
		add_param(_Wxc);
	}
	

	if (_Wxig == nullptr) {
		_Wxig = new Param(IDGen::instance().next(), new Tensor({ _dim, _in_dim }, Tensor::ZERO));
		_initializer->init(_Wxig->get_data());
		add_param(_Wxig);
	}
	

	if (_Wxfg == nullptr) {
		_Wxfg = new Param(IDGen::instance().next(), new Tensor({ _dim, _in_dim }, Tensor::ZERO));
		_initializer->init(_Wxfg->get_data());
		add_param(_Wxfg);
	}
	

	if (_Wxog == nullptr) {
		_Wxog = new Param(IDGen::instance().next(), new Tensor({ _dim, _in_dim }, Tensor::ZERO));
		_initializer->init(_Wxog->get_data());
		add_param(_Wxog);
	}	

	cout << _id << " " << (_input_dim > 0 ? _input_dim : _in_dim - _dim) << " - " << _dim << endl;
}

void LSTMLayer::activate()
{	
	_context = NeuronOperator::init_auxiliary_parameter(_context, _batch_size, _dim);
	_state = NeuronOperator::init_auxiliary_parameter(_state, _batch_size, _dim);
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

	TensorOperator::instance().lstm_state(_batch_size, _state->arr(), _ig->get_output()->arr(), _fg->get_output()->arr(), _cec->get_output()->arr(), _dim);
	Tensor *ac = _activation_function->forward(_state);
	TensorOperator::instance().vv_ewprod(_og->get_output()->arr(), ac->arr(), _output->arr(), _batch_size * _dim);

	_context->override(_output);
}

void LSTMLayer::calc_gradient(Gradient& p_gradient_map, map<string, Tensor*>& p_derivative_map)
{
	BaseLayer::calc_gradient(p_gradient_map, p_derivative_map);

	if (_mode == RTRL)
	{
		Tensor* h = _activation_function->forward(_state);
		Tensor dh = _activation_function->derivative(*_state);

		Tensor*	 df = _activation_function->backward(_delta_out);

		_delta[_og->get_id()] = NeuronOperator::init_auxiliary_parameter(_delta[_og->get_id()], _batch_size, _dim);
		_delta[_cec->get_id()] = NeuronOperator::init_auxiliary_parameter(_delta[_cec->get_id()], _batch_size, _dim);
		_delta[_ig->get_id()] = NeuronOperator::init_auxiliary_parameter(_delta[_ig->get_id()], _batch_size, _dim);
		_delta[_fg->get_id()] = NeuronOperator::init_auxiliary_parameter(_delta[_fg->get_id()], _batch_size, _dim);
		_delta["state"] = NeuronOperator::init_auxiliary_parameter(_delta["state"], _batch_size, _dim);

		TensorOperator::instance().lstm_delta(_batch_size, _delta[_og->get_id()]->arr(), _og->derivative().arr(), h->arr(), df->arr(), _dim);
		TensorOperator::instance().lstm_delta(_batch_size, _delta["state"]->arr(), _og->get_output()->arr(), dh.arr(), df->arr(), _dim);

		_delta[_cec->get_id()]->override(p_derivative_map[_cec->get_bias()->get_id()]);
		TensorOperator::instance().vv_ewprod(_delta["state"]->arr(), p_derivative_map[_ig->get_bias()->get_id()]->arr(), _delta[_ig->get_id()]->arr(), _batch_size * _dim);
		TensorOperator::instance().vv_ewprod(_delta["state"]->arr(), p_derivative_map[_fg->get_bias()->get_id()]->arr(), _delta[_fg->get_id()]->arr(), _batch_size * _dim);

		Tensor* gWxig = &p_gradient_map[_Wxig->get_id()];
		Tensor* gWxfg = &p_gradient_map[_Wxfg->get_id()];
		Tensor* gWxog = &p_gradient_map[_Wxog->get_id()];
		Tensor* gWxc = &p_gradient_map[_Wxc->get_id()];

		Tensor* dWxig = p_derivative_map[_Wxig->get_id()];
		Tensor* dWxfg = p_derivative_map[_Wxfg->get_id()];
		Tensor* dWxc = p_derivative_map[_Wxc->get_id()];

		TensorOperator::instance().full_w_gradient(_batch_size, _input->arr(), _delta[_og->get_id()]->arr(), gWxog->arr(), _dim, _in_dim, false);
		TensorOperator::instance().lstm_w_gradient(_batch_size, gWxig->arr(), _delta["state"]->arr(), dWxig->arr(), _dim, _in_dim);
		TensorOperator::instance().lstm_w_gradient(_batch_size, gWxfg->arr(), _delta["state"]->arr(), dWxfg->arr(), _dim, _in_dim);
		TensorOperator::instance().lstm_w_gradient(_batch_size, gWxc->arr(), _delta["state"]->arr(), dWxc->arr(), _dim, _in_dim);
		TensorOperator::instance().full_b_gradient(_batch_size, _delta[_og->get_id()]->arr(), p_gradient_map[_og->get_bias()->get_id()].arr(), _dim, false);
		TensorOperator::instance().full_b_gradient(_batch_size, _delta[_cec->get_id()]->arr(), p_gradient_map[_cec->get_bias()->get_id()].arr(), _dim, false);
		TensorOperator::instance().full_b_gradient(_batch_size, _delta[_ig->get_id()]->arr(), p_gradient_map[_ig->get_bias()->get_id()].arr(), _dim, false);
		TensorOperator::instance().full_b_gradient(_batch_size, _delta[_fg->get_id()]->arr(), p_gradient_map[_fg->get_bias()->get_id()].arr(), _dim, false);
	}

	Tensor*	 delta_in = nullptr;
	Tensor*	 delta = nullptr;
	if (!_input_layer.empty())
	{
		delta_in = NeuronOperator::init_auxiliary_parameter(delta_in, _batch_size, _in_dim);
		delta = NeuronOperator::init_auxiliary_parameter(delta, _batch_size, _in_dim);

		TensorOperator::instance().full_delta(_batch_size, delta->arr(), _delta[_cec->get_id()]->arr(), _Wxc->get_data()->arr(), _dim, _in_dim);
		TensorOperator::instance().vv_add(delta->arr(), delta_in->arr(), delta_in->arr(), _batch_size *_in_dim);
		TensorOperator::instance().full_delta(_batch_size, delta->arr(), _delta[_ig->get_id()]->arr(), _Wxig->get_data()->arr(), _dim, _in_dim);
		TensorOperator::instance().vv_add(delta->arr(), delta_in->arr(), delta_in->arr(), _batch_size *_in_dim);
		TensorOperator::instance().full_delta(_batch_size, delta->arr(), _delta[_fg->get_id()]->arr(), _Wxfg->get_data()->arr(), _dim, _in_dim);
		TensorOperator::instance().vv_add(delta->arr(), delta_in->arr(), delta_in->arr(), _batch_size *_in_dim);
		TensorOperator::instance().full_delta(_batch_size, delta->arr(), _delta[_og->get_id()]->arr(), _Wxog->get_data()->arr(), _dim, _in_dim);
		TensorOperator::instance().vv_add(delta->arr(), delta_in->arr(), delta_in->arr(), _batch_size *_in_dim);

		int index = _input_dim;

		for (auto it : _input_layer)
		{
			_delta_in[it->get_id()] = NeuronOperator::init_auxiliary_parameter(_delta_in[it->get_id()], _batch_size, it->get_dim());
			delta_in->splice(index, _delta_in[it->get_id()]);
			index += it->get_dim();
		}

		delete delta_in;
		delete delta;
	}
}

void LSTMLayer::calc_derivative(map<string, Tensor*>& p_derivative)
{
	Tensor dcec = _cec->derivative();
	Tensor dig = _ig->derivative();
	Tensor dfg = _fg->derivative();

	p_derivative[_Wxc->get_id()] = NeuronOperator::init_auxiliary_parameter(p_derivative[_Wxc->get_id()], _batch_size, _dim * _in_dim);
	p_derivative[_Wxfg->get_id()] = NeuronOperator::init_auxiliary_parameter(p_derivative[_Wxfg->get_id()], _batch_size, _dim * _in_dim);
	p_derivative[_Wxig->get_id()] = NeuronOperator::init_auxiliary_parameter(p_derivative[_Wxig->get_id()], _batch_size, _dim * _in_dim);
	p_derivative[_cec->get_bias()->get_id()] = NeuronOperator::init_auxiliary_parameter(p_derivative[_cec->get_bias()->get_id()], _batch_size, _dim);
	p_derivative[_fg->get_bias()->get_id()] = NeuronOperator::init_auxiliary_parameter(p_derivative[_fg->get_bias()->get_id()], _batch_size, _dim);
	p_derivative[_ig->get_bias()->get_id()] = NeuronOperator::init_auxiliary_parameter(p_derivative[_ig->get_bias()->get_id()], _batch_size, _dim);

	Tensor bias_input = Tensor::Ones({ _batch_size, 1 });

	TensorOperator::instance().lstm_derivative(_batch_size, p_derivative[_Wxc->get_id()]->arr(), _fg->get_output()->arr(), dcec.arr(), _ig->get_output()->arr(), _input->arr(), _dim, _in_dim);
	TensorOperator::instance().lstm_derivative(_batch_size, p_derivative[_Wxig->get_id()]->arr(), _fg->get_output()->arr(), _cec->get_output()->arr(), dig.arr(), _input->arr(), _dim, _in_dim);
	TensorOperator::instance().lstm_derivative(_batch_size, p_derivative[_Wxfg->get_id()]->arr(), _fg->get_output()->arr(), _state->arr(), dfg.arr(), _input->arr(), _dim, _in_dim);

	TensorOperator::instance().lstm_derivative(_batch_size, p_derivative[_cec->get_bias()->get_id()]->arr(), _fg->get_output()->arr(), dcec.arr(), _ig->get_output()->arr(), bias_input.arr(), _dim, 1);
	TensorOperator::instance().lstm_derivative(_batch_size, p_derivative[_ig->get_bias()->get_id()]->arr(), _fg->get_output()->arr(), _cec->get_output()->arr(), dig.arr(), bias_input.arr(), _dim, 1);
	TensorOperator::instance().lstm_derivative(_batch_size, p_derivative[_fg->get_bias()->get_id()]->arr(), _fg->get_output()->arr(), _state->arr(), dfg.arr(), bias_input.arr(), _dim, 1);
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

Tensor* LSTMLayer::get_dim_tensor()
{
	if (_dim_tensor == nullptr)
	{
		_dim_tensor = new Tensor({ 1 }, Tensor::VALUE, _dim);
	}

	return _dim_tensor;
}
