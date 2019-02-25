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
	_partial_deriv[l->_input_group->get_id() + " " + l->_cec->get_id()] = Tensor::Zero({ l->_cec->get_dim(), l->_input_group->get_dim() });
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

	p_gradient[l->_in_output_gate->get_id()] = _delta[l->_output_gate->get_id()].outer_prod(*l->_aux_input->get_output());

	float* dwf = &p_gradient[l->_in_forget_gate->get_id()].arr()[0];
	float* dwi = &p_gradient[l->_in_input_gate->get_id()].arr()[0];

	float* pd_in_forget_gate = &_partial_deriv[l->_in_forget_gate->get_id()].arr()[0];
	float* pd_in_input_gate = &_partial_deriv[l->_in_input_gate->get_id()].arr()[0];
	float* state_error = &_state_error.arr()[0];

	for (int j = 0; j < l->_cec->get_dim(); j++)
	{
		for (int m = 0; m < l->_aux_input->get_dim(); m++)
		{
			*dwf++ = *state_error * *pd_in_forget_gate++;
			*dwi++ = *state_error * *pd_in_input_gate++;
		}
		state_error++;
	}
	
	float* pd_in_cec = &_partial_deriv[l->_input_group->get_id() + " " + l->_cec->get_id()].arr()[0];

	vector<BaseLayer*> input_layers = _network->get_input_layers(_layer->get_id());

	for (auto it = input_layers.begin(); it != input_layers.end(); ++it) {
		Connection* c = _network->get_connection((*it)->get_id(), _layer->get_id());
		if (c->is_trainable())
		{
			float* dwc = &p_gradient[c->get_id()].arr()[0];
			state_error = &_state_error.arr()[0];

			for (int j = 0; j < c->get_out_dim(); j++)
			{
				for (int m = 0; m < c->get_in_dim(); m++)
				{
					*dwc++ = *state_error * *pd_in_cec++;
				}
				state_error++;
			}
		}
	}

	float* dwcc = &p_gradient[l->_ct_cec->get_id()].arr()[0];
	float* pd_ct_cec = &_partial_deriv[l->_ct_cec->get_id()].arr()[0];
	state_error = &_state_error.arr()[0];

	for (int j = 0; j < l->_ct_cec->get_out_dim(); j++)
	{
		for (int m = 0; m < l->_ct_cec->get_in_dim(); m++)
		{
			*dwcc++ = *state_error * *pd_ct_cec++;
		}
		state_error++;
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
	float* gx = &g.arr()[0];
	const Tensor h = l->_cec->get_h();
	float* hx = &h.arr()[0];
	const Tensor dg = l->_cec->get_dg();
	float* dgx = &dg.arr()[0];

	float d0;
	float d1;

	float* pdiigx = &_partial_deriv[l->_in_input_gate->get_id()].arr()[0];
	float* pdifgx = &_partial_deriv[l->_in_forget_gate->get_id()].arr()[0];
	float* pdicx = &_partial_deriv[l->_input_group->get_id() + " " + l->_cec->get_id()].arr()[0];
	float* pdccx = &_partial_deriv[l->_ct_cec->get_id()].arr()[0];

	float* pdigx = &_partial_deriv[l->_input_gate->get_id()].arr()[0];
	float* pdfgx = &_partial_deriv[l->_forget_gate->get_id()].arr()[0];
	float* pdcx = &_partial_deriv[l->_cec->get_id()].arr()[0];

	Tensor input_gate_doutput = l->_input_gate->get_deriv_output();
	float* digx = &input_gate_doutput.arr()[0];
	Tensor forget_gate_doutput = l->_forget_gate->get_deriv_output();
	float* dfgx = &forget_gate_doutput.arr()[0];

	float* fgx = &l->_forget_gate->get_output()->arr()[0];
	float* igx = &l->_input_gate->get_output()->arr()[0];

	for (int j = 0; j < l->_cec->get_dim(); j++)
	{
		float* ax = &l->_aux_input->get_output()->arr()[0];
		float* ix = &l->_input_group->get_output()->arr()[0];
		float* cx = &l->_context->get_output()->arr()[0];

		for (int m = 0; m < l->_aux_input->get_dim(); m++)
		{
			d0 = *pdiigx * *fgx;
			d1 = *gx * *digx * *ax;
			*pdiigx++ = d0 + d1;

			d0 = *pdifgx * *fgx;
			d1 = *hx * *dfgx * *ax;
			*pdifgx++ = d0 + d1;
			ax++;
		}

		for (int m = 0; m < l->_input_group->get_dim(); m++)
		{
			d0 = *pdicx * *fgx;
			d1 = *dgx * *igx * *ix++;
			*pdicx++ = d0 + d1;
		}

		for (int m = 0; m < l->_context->get_dim(); m++)
		{
			d0 = *pdccx * *fgx;
			d1 = *dgx * *igx * *cx++;
			*pdccx++ = d0 + d1;
		}

		d0 = *pdigx * *fgx;
		d1 = *gx * *digx++;
		*pdigx++ = d0 + d1;

		d0 = *pdfgx * *fgx;
		d1 = *gx * *dfgx++;
		*pdfgx++ = d0 + d1;

		d0 = *pdcx * *fgx;
		d1 = *dgx * *igx;
		*pdcx++ = d0 + d1;

		gx++;
		hx++;
		dgx++;
		fgx++;
		igx++;
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
