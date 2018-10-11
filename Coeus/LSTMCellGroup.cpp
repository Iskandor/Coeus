#include "LSTMCellGroup.h"

using namespace Coeus;

LSTMCellGroup::LSTMCellGroup(int p_dim, ACTIVATION p_activation_function, SimpleCellGroup* p_input_gate, SimpleCellGroup* p_output_gate, SimpleCellGroup* p_forget_gate) : BaseCellGroup(p_dim, true)
{
	_state = Tensor::Zero({p_dim});

	_f = init_activation_function(p_activation_function);
	_g = init_activation_function(TANH);

	_input_gate = p_input_gate;
	_output_gate = p_output_gate;
	_forget_gate = p_forget_gate;
}

LSTMCellGroup::LSTMCellGroup(nlohmann::json p_data) : BaseCellGroup(p_data)
{
}

LSTMCellGroup::LSTMCellGroup(LSTMCellGroup& p_copy) : BaseCellGroup(p_copy._dim, p_copy._bias_flag)
{
	_state = Tensor::Zero({ p_copy._dim });

	_f = init_activation_function(p_copy._f->get_type());
	_g = init_activation_function(TANH);
	_input_gate = p_copy._input_gate;
	_output_gate = p_copy._output_gate;
	_forget_gate = p_copy._forget_gate;

}

LSTMCellGroup& LSTMCellGroup::operator=(const LSTMCellGroup& p_copy)
{
	copy(p_copy);

	_state = Tensor::Zero({ p_copy._dim });
	_f = init_activation_function(p_copy._f->get_type());
	_g = init_activation_function(SIGMOID);
	_input_gate = p_copy._input_gate;
	_output_gate = p_copy._output_gate;

	return *this;
}

LSTMCellGroup::~LSTMCellGroup()
{
	delete _g;
}

void LSTMCellGroup::integrate(Tensor* p_input, Tensor* p_weights)
{
	_net += *p_weights * *p_input;
}

void LSTMCellGroup::activate()
{
	activate(_input_gate->get_output(), _output_gate->get_output(), _forget_gate->get_output());
}

void LSTMCellGroup::reset() const
{
	_state.fill(0);
}

LSTMCellGroup* LSTMCellGroup::clone()
{
	return new LSTMCellGroup(*this);
}

void LSTMCellGroup::activate(Tensor* p_input_gate, Tensor* p_output_gate, Tensor* p_forget_gate)
{
	if (is_bias()) {
		_net += _bias;
	}

	_g_output = _g->activate(_net);
	_state = _state.dot(*p_forget_gate) + _g_output.dot(*p_input_gate);

	_h_output = _f->activate(_state);
	_output = _h_output.dot(*p_output_gate);

	_net.fill(0);
}

Tensor LSTMCellGroup::get_h() const
{
	return _h_output;
}

Tensor LSTMCellGroup::get_dh() {
	return _f->deriv(_h_output);
}

Tensor LSTMCellGroup::get_g() const
{
	return _g_output;
}

Tensor LSTMCellGroup::get_dg() {
	return _g->deriv(_g_output);
}
