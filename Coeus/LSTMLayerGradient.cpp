#include "LSTMLayerGradient.h"
#include "ActivationFunctionsDeriv.h"
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

	_dc_input_gate = Tensor::Zero({ l->_input_gate->get_dim(), l->_aux_input->get_dim() });
	_dc_forget_gate = Tensor::Zero({ l->_input_gate->get_dim(), l->_aux_input->get_dim() });
}

void LSTMLayerGradient::calc_deriv()
{
	LSTMLayer* l = get_layer<LSTMLayer>();

	calc_deriv_group(l->_output_gate);
	calc_deriv_group(l->_input_gate);
	calc_deriv_group(l->_forget_gate);
}

void LSTMLayerGradient::calc_delta(Tensor* p_weights, Tensor* p_delta)
{
	LSTMLayer* l = get_layer<LSTMLayer>();
	const Tensor h = l->get_output_group<LSTMCellGroup>()->get_h();
	const Tensor dh = l->get_output_group<LSTMCellGroup>()->get_dh();
	const Tensor dg = l->get_output_group<LSTMCellGroup>()->get_dg();

	_delta[l->_output_gate->get_id()] = _deriv[l->_output_gate->get_id()].dot(h) * (p_weights->T() * *p_delta);

	_state_error = l->_output_gate->get_output()->dot(dh) * (p_weights->T() * *p_delta);


}

void LSTMLayerGradient::calc_gradient(map<string, Tensor>& p_w_gradient, map<string, Tensor>& p_b_gradient)
{
	LSTMLayer* l = get_layer<LSTMLayer>();

	p_w_gradient[l->_output_gate->get_id()] = _delta[l->_output_gate->get_id()] * *l->_aux_input->get_output();
}
