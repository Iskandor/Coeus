#include "LSTMLayer.h"

using namespace Coeus;


LSTMLayer::LSTMLayer(const string p_id, int p_dim, NeuralGroup::ACTIVATION p_activation) : BaseLayer(p_id)
{
	_type = LSTM;

	_output_group = new NeuralGroup(p_dim, p_activation, true);

	_hf = new NeuralGroup(p_dim, NeuralGroup::SIGMOID, true);
	_hi = new NeuralGroup(p_dim, NeuralGroup::SIGMOID, true);
	_ho = new NeuralGroup(p_dim, NeuralGroup::SIGMOID, true);
	_hc = new NeuralGroup(p_dim, NeuralGroup::TANH, true);

	_c = new Tensor({ p_dim }, Tensor::INIT::ZERO);
	_h = new Tensor({ p_dim }, Tensor::INIT::ZERO);
	_c_old = new Tensor({ p_dim }, Tensor::INIT::ZERO);
	_h_old = new Tensor({ p_dim }, Tensor::INIT::ZERO);
}

LSTMLayer::~LSTMLayer()
{
	delete _input_group;
	delete _output_group;
	delete _x;
	delete _c;
	delete _h;
	delete _c_old;
	delete _h_old;
	delete _hf;
	delete _hi;
	delete _ho;
	delete _hc;
	delete _Wf;
	delete _Wi;
	delete _Wo;
	delete _Wc;
}

void LSTMLayer::integrate(Tensor* p_input, Tensor* p_weights)
{
	_x->override(p_input);
	Tensor::Concat(_input_group->get_output(), _h_old, p_input);
}

void LSTMLayer::activate(Tensor* p_input)
{
	_hf->integrate(_input_group->get_output(), _Wf->get_weights());
	_hf->activate();
	_hi->integrate(_input_group->get_output(), _Wi->get_weights());
	_hi->activate();
	_ho->integrate(_input_group->get_output(), _Wo->get_weights());
	_ho->activate();
	_hc->integrate(_input_group->get_output(), _Wc->get_weights());
	_hc->activate();

	for(int i = 0; i < _c->size(); i++)
	{
		_c->set(i, _hf->get_output()->at(i) * _c->at(i) + _hi->get_output()->at(i) * _hc->get_output()->at(i));
		_h->set(i, _ho->get_output()->at(i) * tanh(_c->at(i)));
	}

	_output_group->integrate(_h, _Wy->get_weights());
	_output_group->activate();

	_c_old->override(_c);
	_h_old->override(_h);
}

void LSTMLayer::override_params(BaseLayer* p_source)
{
//#TODO doplnit prepis parametrov LSTM siete
}

void LSTMLayer::post_connection(BaseLayer* p_input)
{
	const int dim = output_dim() + p_input->output_dim();

	_input_group = new NeuralGroup(dim, NeuralGroup::LINEAR, false);

	_x = new Tensor({ p_input->output_dim() }, Tensor::ZERO);
	_Wf = add_connection(new Connection(dim, output_dim(), _input_group->get_id(), _hf->get_id()));
	_Wf->init(Connection::INIT::GLOROT_UNIFORM);
	_Wi = add_connection(new Connection(dim, output_dim(), _input_group->get_id(), _hi->get_id()));
	_Wi->init(Connection::INIT::GLOROT_UNIFORM);
	_Wo = add_connection(new Connection(dim, output_dim(), _input_group->get_id(), _ho->get_id()));
	_Wo->init(Connection::INIT::GLOROT_UNIFORM);
	_Wc = add_connection(new Connection(dim, output_dim(), _input_group->get_id(), _hc->get_id()));
	_Wc->init(Connection::INIT::GLOROT_UNIFORM);
	_Wy = add_connection(new Connection(output_dim(), output_dim(), "_h", _output_group->get_id()));
	_Wy->init(Connection::INIT::GLOROT_UNIFORM);
}
