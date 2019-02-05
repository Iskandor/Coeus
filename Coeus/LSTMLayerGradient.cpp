#include "LSTMLayerGradient.h"

using namespace Coeus;

LSTMLayerGradient::LSTMLayerGradient(BaseLayer* p_layer, NeuralNetwork* p_network) : IGradientComponent(p_layer, p_network)
{
	
}


LSTMLayerGradient::~LSTMLayerGradient()
{
}

void LSTMLayerGradient::init()
{
	auto* l = get_layer<LSTMLayer>();

	_state_error = Tensor::Zero({ l->_cec->get_dim() });

	_partial_deriv[l->_in_input_gate->get_id()] = Tensor::Zero({ l->_cec->get_dim(), l->_aux_input->get_dim() });
	_partial_deriv[l->_in_forget_gate->get_id()] = Tensor::Zero({ l->_cec->get_dim(), l->_aux_input->get_dim() });
	_partial_deriv[l->_input_group->get_id() + " " + l->_cec->get_id()] = Tensor::Zero({ l->_cec->get_dim(), l->_cec->get_dim() });
	_partial_deriv[l->_ct_cec->get_id()] = Tensor::Zero({ l->_cec->get_dim(), l->_cec->get_dim() });

	_partial_deriv[l->_cec->get_id()] = Tensor::Zero({ l->_cec->get_dim() });
	_partial_deriv[l->_input_gate->get_id()] = Tensor::Zero({ l->_cec->get_dim() });
	_partial_deriv[l->_forget_gate->get_id()] = Tensor::Zero({ l->_cec->get_dim() });
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

	_delta[l->_input_group->get_id()] = _state_error;
}

void LSTMLayerGradient::calc_gradient(map<string, Tensor>& p_gradient)
{
	LSTMLayer* l = get_layer<LSTMLayer>();

	p_gradient[l->_in_output_gate->get_id()] = _delta[l->_output_gate->get_id()] * *l->_aux_input->get_output();

	Tensor* dwf = &p_gradient[l->_in_forget_gate->get_id()];
	Tensor* dwi = &p_gradient[l->_in_input_gate->get_id()];

	Tensor* pd_in_forget_gate = &_partial_deriv[l->_in_forget_gate->get_id()];
	Tensor* pd_in_input_gate = &_partial_deriv[l->_in_input_gate->get_id()];

	for (int j = 0; j < l->_cec->get_dim(); j++)
	{
		for (int m = 0; m < l->_aux_input->get_dim(); m++)
		{
			dwf->set(j, m, _state_error[j] * pd_in_forget_gate->at(j, m));
			dwi->set(j, m, _state_error[j] * pd_in_input_gate->at(j, m));
		}
	}
	
	Tensor* pd_in_cec = &_partial_deriv[l->_input_group->get_id() + " " + l->_cec->get_id()];

	vector<BaseLayer*> input_layers = _network->get_input_layers(_layer->get_id());

	for (auto it = input_layers.begin(); it != input_layers.end(); ++it) {
		Connection* c = _network->get_connection((*it)->get_id(), _layer->get_id());
		if (c->is_trainable())
		{
			Tensor* dwc = &p_gradient[c->get_id()];
			for (int j = 0; j < c->get_out_dim(); j++)
			{
				for (int m = 0; m < c->get_in_dim(); m++)
				{
					dwc->set(j, m, _state_error[j] * pd_in_cec->at(j, m));
				}
			}
		}
	}

	Tensor* dwcc = &p_gradient[l->_ct_cec->get_id()];
	Tensor* pd_ct_cec = &_partial_deriv[l->_ct_cec->get_id()];

	for (int j = 0; j < l->_ct_cec->get_out_dim(); j++)
	{
		for (int m = 0; m < l->_ct_cec->get_in_dim(); m++)
		{
			dwcc->set(j, m, _state_error[j] * pd_ct_cec->at(j, m));
		}
	}

	p_gradient[l->_cec->get_id()] = _partial_deriv[l->_cec->get_id()];
	p_gradient[l->_input_gate->get_id()] = _state_error.dot(_partial_deriv[l->_input_gate->get_id()]);
	p_gradient[l->_forget_gate->get_id()] = _state_error.dot(_partial_deriv[l->_forget_gate->get_id()]);
	p_gradient[l->_output_gate->get_id()] = _delta[l->_output_gate->get_id()];
}

void LSTMLayerGradient::calc_deriv_estimate()
{
	LSTMLayer* l = get_layer<LSTMLayer>();

	const Tensor g = l->_cec->get_g();
	const Tensor h = l->_cec->get_h();
	const Tensor dg = l->_cec->get_dg();

	double d0;
	double d1;

	Tensor* pd_in_input_gate = &_partial_deriv[l->_in_input_gate->get_id()];
	Tensor* pd_in_forget_gate = &_partial_deriv[l->_in_forget_gate->get_id()];
	Tensor* pd_in_cec = &_partial_deriv[l->_input_group->get_id() + " " + l->_cec->get_id()];
	Tensor* pd_ct_cec = &_partial_deriv[l->_ct_cec->get_id()];

	Tensor* pd_input_gate = &_partial_deriv[l->_input_gate->get_id()];
	Tensor* pd_forget_gate = &_partial_deriv[l->_forget_gate->get_id()];
	Tensor* pd_cec = &_partial_deriv[l->_cec->get_id()];

	Tensor input_gate_doutput = l->_input_gate->get_deriv_output();
	Tensor forget_gate_doutput = l->_forget_gate->get_deriv_output();

	for (int j = 0; j < l->_cec->get_dim(); j++)
	{
		for (int m = 0; m < l->_aux_input->get_dim(); m++)
		{
			d0 = pd_in_input_gate->at(j, m) * l->_forget_gate->get_output()->at(j);
			d1 = g[j] * input_gate_doutput.at(j, j) * l->_aux_input->get_output()->at(m);
			pd_in_input_gate->set(j, m, d0 + d1);

			d0 = pd_in_forget_gate->at(j, m) * l->_forget_gate->get_output()->at(j);
			d1 = h[j] * forget_gate_doutput.at(j, j) * l->_aux_input->get_output()->at(m);
			pd_in_forget_gate->set(j, m, d0 + d1);
		}

		for (int m = 0; m < l->_input_group->get_dim(); m++)
		{
			d0 = pd_in_cec->at(j, m) * l->_forget_gate->get_output()->at(j);
			d1 = dg[j] * l->_input_gate->get_output()->at(j) * l->_input_group->get_output()->at(m);
			pd_in_cec->set(j, m, d0 + d1);
		}

		for (int m = 0; m < l->_context->get_dim(); m++)
		{
			d0 = pd_ct_cec->at(j, m) * l->_forget_gate->get_output()->at(j);
			d1 = dg[j] * l->_input_gate->get_output()->at(j) * l->_context->get_output()->at(m);
			pd_ct_cec->set(j, m, d0 + d1);
		}

		d0 = (*pd_input_gate)[j] * l->_forget_gate->get_output()->at(j);
		d1 = g[j] * input_gate_doutput.at(j, j);
		(*pd_input_gate)[j] = d0 + d1;

		d0 = (*pd_forget_gate)[j] * l->_forget_gate->get_output()->at(j);
		d1 = g[j] * forget_gate_doutput.at(j, j);
		(*pd_forget_gate)[j] = d0 + d1;

		d0 = (*pd_cec)[j] * l->_forget_gate->get_output()->at(j);
		d1 = dg[j] * l->_input_gate->get_output()->at(j);
		(*pd_cec)[j] = d0 + d1;
	}
}

void LSTMLayerGradient::reset()
{
	LSTMLayer* l = get_layer<LSTMLayer>();
	_partial_deriv[l->_in_input_gate->get_id()].fill(0);
	_partial_deriv[l->_in_forget_gate->get_id()].fill(0);
	_partial_deriv[l->_input_group->get_id() + " " + l->_cec->get_id()].fill(0);
	_partial_deriv[l->_ct_cec->get_id()].fill(0);

	_partial_deriv[l->_input_gate->get_id()].fill(0);
	_partial_deriv[l->_forget_gate->get_id()].fill(0);
	_partial_deriv[l->_cec->get_id()].fill(0);

	l->reset();
}
