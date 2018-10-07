#include "LSTMLayerGradient.h"
#include "LSTMLayerState.h"

using namespace Coeus;

LSTMLayerGradient::LSTMLayerGradient(BaseLayer* p_layer, NeuralNetwork* p_network) : IGradientComponent(p_layer, p_network)
{

}


LSTMLayerGradient::~LSTMLayerGradient()
{
}

void LSTMLayerGradient::init()
{
	_state_error = Tensor::Zero({ _layer->get_output_group<LSTMCellGroup>()->get_dim() });

	LSTMLayer* l = get_layer<LSTMLayer>();

	_dc_input_gate = Tensor::Zero({ l->_input_gate->get_dim() });
	_dc_forget_gate = Tensor::Zero({ l->_forget_gate->get_dim() });
	_dc_input = Tensor::Zero({ l->_aux_input->get_dim() });
}

void LSTMLayerGradient::calc_deriv()
{
	LSTMLayer* l = get_layer<LSTMLayer>();

	calc_deriv_group(l->_output_gate);
	calc_deriv_group(l->_input_gate);
	calc_deriv_group(l->_forget_gate);

	const Tensor g = l->_cec->get_g();
	const Tensor dg = l->_cec->get_dg();
	const Tensor h = l->_cec->get_h();

	_dc_input = _dc_input.dot(*l->_forget_gate->get_output()) + dg * *l->_input_gate->get_output() * *l->_aux_input->get_output();
	_dc_input_gate = _dc_input_gate.dot(*l->_forget_gate->get_output()) + g * _deriv[l->_input_gate->get_id()] * *l->_aux_input->get_output();
	_dc_forget_gate = _dc_forget_gate.dot(*l->_forget_gate->get_output()) + h * _deriv[l->_forget_gate->get_id()] * *l->_aux_input->get_output();
}

void LSTMLayerGradient::calc_delta(Tensor* p_weights, Tensor* p_delta)
{
	LSTMLayer* l = get_layer<LSTMLayer>();
	const Tensor h = l->_cec->get_h();
	const Tensor dh = l->_cec->get_dh();

	_delta[l->_output_gate->get_id()] = _deriv[l->_output_gate->get_id()].dot(h) * (p_weights->T() * *p_delta);

	_state_error = l->_output_gate->get_output()->dot(dh) * (p_weights->T() * *p_delta);

	 
}

void LSTMLayerGradient::calc_gradient(map<string, Tensor>& p_w_gradient, map<string, Tensor>& p_b_gradient)
{
	LSTMLayer* l = get_layer<LSTMLayer>();

	p_w_gradient[l->_output_gate->get_id()] = _delta[l->_output_gate->get_id()] * *l->_aux_input->get_output();
}