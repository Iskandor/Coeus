#include "GRULayer.h"
#include "ActivationFunctionFactory.h"
#include "IDGen.h"
#include "TensorOperator.h"

using namespace Coeus;

GRULayer::GRULayer(const string& p_id, const int p_dim, const ACTIVATION p_activation, TensorInitializer* p_initializer, const int p_in_dim) : BaseLayer(p_id, p_dim, {p_in_dim})
{
	_type = GRU;
	_is_recurrent = true;

	_initializer = p_initializer;

	_y = new NeuronOperator(p_dim, p_activation);
	add_param(_y);
	_h = new NeuronOperator(p_dim, TANH);
	add_param(_h);
	_rg = new NeuronOperator(p_dim, SIGMOID);
	add_param(_rg);
	_ug = new NeuronOperator(p_dim, SIGMOID);
	add_param(_ug);

	_Why = nullptr;
	_Wxh = nullptr;
	_Wxrg = nullptr;
	_Wxug = nullptr;

	_reseted_input = nullptr;
	_h_input = nullptr;
}


GRULayer::~GRULayer()
{
	delete _y;
	delete _h;
	delete _rg;
	delete _ug;

	delete _Why;
	delete _Wxh;
	delete _Wxrg;
	delete _Wxug;

	delete _reseted_input;
	delete _h_input;
}

BaseLayer* GRULayer::clone()
{
	return nullptr;
}

void GRULayer::init(vector<BaseLayer*>& p_input_layers, vector<BaseLayer*>& p_output_layers)
{
	BaseLayer::init(p_input_layers, p_output_layers);

	if (_Why == nullptr) {
		_Why = new Param(IDGen::instance().next(), new Tensor({ _dim, _in_dim }, Tensor::ZERO));
		_initializer->init(_Why->get_data());
	}
	add_param(_Why->get_id(), _Why->get_data());


	_in_dim += _dim;

	if (_Wxh == nullptr) {
		_Wxh = new Param(IDGen::instance().next(), new Tensor({ _dim, _in_dim }, Tensor::ZERO));
		_initializer->init(_Wxh->get_data());
	}
	add_param(_Wxh->get_id(), _Wxh->get_data());

	if (_Wxrg == nullptr) {
		_Wxrg = new Param(IDGen::instance().next(), new Tensor({ _dim, _in_dim }, Tensor::ZERO));
		_initializer->init(_Wxrg->get_data());
	}
	add_param(_Wxrg->get_id(), _Wxrg->get_data());

	if (_Wxug == nullptr) {
		_Wxug = new Param(IDGen::instance().next(), new Tensor({ _dim, _in_dim }, Tensor::ZERO));
		_initializer->init(_Wxug->get_data());
	}
	add_param(_Wxug->get_id(), _Wxug->get_data());

	cout << _id << " " << (_input_dim > 0 ? _input_dim : _in_dim - _dim) << " - " << _dim << endl;
}

void GRULayer::integrate(Tensor* p_input)
{
	BaseLayer::integrate(p_input);
	_h_input = NeuronOperator::init_auxiliary_parameter(_h_input, _batch_size, _in_dim);
	_h_input->push_back(p_input);
}

void GRULayer::activate()
{
	_reseted_input = NeuronOperator::init_auxiliary_parameter(_reseted_input, _batch_size, _dim);
	
	_state = NeuronOperator::init_auxiliary_parameter(_state, _batch_size, _dim);

	_input->push_back(_state);
	_input->reset_index();

	_rg->integrate(_input, _Wxrg->get_data());
	_rg->activate();

	_ug->integrate(_input, _Wxug->get_data());
	_ug->activate();

	TensorOperator::instance().vv_ewprod(_rg->get_output()->arr(), _state->arr(), _reseted_input->arr(), _batch_size * _dim);

	_h_input->push_back(_reseted_input);
	_h_input->reset_index();

	_h->integrate(_h_input, _Wxh->get_data());
	_h->activate();

	TensorOperator::instance().gru_state(_batch, _state->arr(), _ug->get_output()->arr(), _h->get_output()->arr(), _dim);

	_y->integrate(_state, _Why->get_data());
	_y->activate();

	_output =  _y->get_output();
}

void GRULayer::calc_derivative(map<string, Tensor*>& p_derivative)
{

}

void GRULayer::calc_gradient(Gradient& p_gradient_map, map<string, Tensor*>& p_derivative_map)
{
}

void GRULayer::override(BaseLayer* p_source)
{
}

void GRULayer::reset()
{
	if (_output != nullptr) _output->fill(0);
}

Tensor* GRULayer::get_dim_tensor()
{
	if (_dim_tensor == nullptr)
	{
		_dim_tensor = new Tensor({ 1 }, Tensor::VALUE, _dim);
	}

	return _dim_tensor;
}

json GRULayer::get_json() const
{
	json data = BaseLayer::get_json();

	return data;
}
