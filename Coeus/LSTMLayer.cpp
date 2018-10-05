#include "LSTMLayer.h"
#include "LSTMLayerGradient.h"

using namespace Coeus;


LSTMLayer::LSTMLayer(const string& p_id, int p_dim, ACTIVATION p_activation) : BaseLayer(p_id)
{
	_type = LSTM;

	_input_gate = add_group<SimpleCellGroup>(new SimpleCellGroup(p_dim, SIGMOID, false));
	_output_gate = add_group<SimpleCellGroup>(new SimpleCellGroup(p_dim, SIGMOID, false));
	_forget_gate = add_group<SimpleCellGroup>(new SimpleCellGroup(p_dim, SIGMOID, false));
	_cec = add_group<LSTMCellGroup>(new LSTMCellGroup(p_dim, p_activation, _input_gate, _output_gate, _forget_gate));
	_input_group = _output_group = _cec;

	_gradient_component = new LSTMLayerGradient(this);
}

LSTMLayer::~LSTMLayer()
{
	delete _cec;
	delete _input_gate;
	delete _output_gate;
	delete _forget_gate;
	delete _aux_input;
	delete _in_input_gate;
	delete _in_output_gate;
	delete _in_forget_gate;
	delete _gradient_component;
}

void LSTMLayer::init(vector<BaseLayer*>& p_input_layers)
{
	int dim = 0;
	for (auto& layer : p_input_layers)
	{
		dim += layer->get_output()->size();
	}

	_aux_input = add_group<SimpleCellGroup>(new SimpleCellGroup(dim, LINEAR, false));
	_in_input_gate = add_connection(new Connection(_aux_input->get_dim(), _input_gate->get_dim(), _aux_input->get_id(), _input_gate->get_id()));
	_in_input_gate->init(Connection::LECUN_UNIFORM);
	_in_output_gate = add_connection(new Connection(_aux_input->get_dim(), _output_gate->get_dim(), _aux_input->get_id(), _output_gate->get_id()));
	_in_output_gate->init(Connection::LECUN_UNIFORM);
	_in_forget_gate = add_connection(new Connection(_aux_input->get_dim(), _forget_gate->get_dim(), _aux_input->get_id(), _forget_gate->get_id()));
	_in_forget_gate->init(Connection::LECUN_UNIFORM);
}

void LSTMLayer::integrate(Tensor* p_input, Tensor* p_weights)
{
	_cec->integrate(p_input, p_weights);
	_input.emplace_back(p_input);
}

void LSTMLayer::activate(Tensor* p_input)
{
	_aux_input->set_output(_input);

	_input_gate->integrate(_aux_input->get_output(), _in_input_gate->get_weights());
	_input_gate->activate();
	_output_gate->integrate(_aux_input->get_output(), _in_output_gate->get_weights());
	_output_gate->activate();
	_forget_gate->integrate(_aux_input->get_output(), _in_forget_gate->get_weights());
	_forget_gate->activate();

	_cec->activate();

	_input.clear();
}

void LSTMLayer::override(BaseLayer* p_source)
{
//#TODO doplnit prepis parametrov LSTM siete
}