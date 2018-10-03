#include "LSTMCellGroup.h"

using namespace Coeus;

LSTMCellGroup::LSTMCellGroup(int p_dim, ACTIVATION p_activation_function, SimpleCellGroup* p_input_gate, SimpleCellGroup* p_output_gate) : BaseCellGroup(p_dim)
{
	_state = Tensor::Zero({p_dim});

	_activation_function = p_activation_function;
	_h = init_activation_function(p_activation_function);
	_g = init_activation_function(SIGMOID);

	_input_gate = p_input_gate;
	_output_gate = p_output_gate;
}

LSTMCellGroup::LSTMCellGroup(nlohmann::json p_data) : BaseCellGroup(p_data)
{
}

LSTMCellGroup::LSTMCellGroup(LSTMCellGroup& p_copy) : BaseCellGroup(p_copy._dim)
{
	_state = Tensor::Zero({ p_copy._dim });

	_activation_function = p_copy._activation_function;
	_h = init_activation_function(p_copy._activation_function);
	_g = init_activation_function(SIGMOID);
	_input_gate = p_copy._input_gate;
	_output_gate = p_copy._output_gate;

}

LSTMCellGroup& LSTMCellGroup::operator=(const LSTMCellGroup& p_copy)
{
	copy(p_copy);

	_state = Tensor::Zero({ p_copy._dim });
	_activation_function = p_copy._activation_function;
	_h = init_activation_function(p_copy._activation_function);
	_g = init_activation_function(SIGMOID);
	_input_gate = p_copy._input_gate;
	_output_gate = p_copy._output_gate;

	return *this;
}

LSTMCellGroup::~LSTMCellGroup()
{
	delete _h;
	delete _g;
}

void LSTMCellGroup::integrate(Tensor* p_input, Tensor* p_weights)
{
	_net += *p_weights * *p_input;
}

void LSTMCellGroup::activate()
{
	activate(_input_gate->get_output(), _output_gate->get_output());
}

void LSTMCellGroup::activate(Tensor* p_input_gate, Tensor* p_output_gate)
{
	Tensor g = _g->activate(_net);
	_state = _state + Tensor::apply(*p_input_gate, g, Tensor::ew_dot);

	Tensor h = _h->activate(_state);
	_output = Tensor::apply(*p_output_gate, h, Tensor::ew_dot);
}
