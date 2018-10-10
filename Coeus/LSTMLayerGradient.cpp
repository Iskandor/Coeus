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
	const Tensor h = l->_cec->get_h();
	const Tensor dh = l->_cec->get_dh();
	const Tensor dkc = p_weights->T() * *p_delta;

	_delta[l->_output_gate->get_id()] = (_deriv[l->_output_gate->get_id()] * h).dot(dkc);

	_state_error = (dh * *l->_output_gate->get_output()).dot(dkc);
}

void LSTMLayerGradient::calc_gradient(map<string, Tensor>& p_w_gradient, map<string, Tensor>& p_b_gradient)
{
	LSTMLayer* l = get_layer<LSTMLayer>();

	p_w_gradient[l->_in_output_gate->get_id()] = _delta[l->_output_gate->get_id()] * *l->_aux_input->get_output();

	Tensor dwf = Tensor::Zero({ l->_cec->get_dim(), l->_aux_input->get_dim() });
	Tensor dwi = Tensor::Zero({ l->_cec->get_dim(), l->_aux_input->get_dim() });

	for (int j = 0; j < l->_cec->get_dim(); j++)
	{
		for (int m = 0; m < l->_aux_input->get_dim(); m++)
		{
			dwf.set(j, m, _state_error[j] * l->_dc_forget_gate.at(j, m));
			dwi.set(j, m, _state_error[j] * l->_dc_input_gate.at(j, m));
		}
	}

	p_w_gradient[l->_in_forget_gate->get_id()] = dwf;
	p_w_gradient[l->_in_input_gate->get_id()] = dwi;

	vector<BaseLayer*> input_layers = _network->get_input_layers(_layer->get_id());

	for (auto it = input_layers.begin(); it != input_layers.end(); ++it) {
		Connection* c = _network->get_connection((*it)->get_id(), _layer->get_id());
		if (c->is_trainable())
		{
			Tensor dwc = Tensor::Zero({ c->get_out_dim(), c->get_in_dim() });
			for (int j = 0; j < c->get_out_dim(); j++)
			{
				for (int m = 0; m < c->get_in_dim(); m++)
				{
					dwc.set(j, m, _state_error[j] * l->_dc_input.at(j, m));
				}
			}

			p_w_gradient[c->get_id()] = dwc;
		}
	}
}