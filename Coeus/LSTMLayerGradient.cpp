#include "LSTMLayerGradient.h"
#include "ActivationFunctionsDeriv.h"
#include "LSTMLayerState.h"

using namespace Coeus;

LSTMLayerGradient::LSTMLayerGradient(LSTMLayer* p_layer) : IGradientComponent(p_layer)
{
	_state = new LSTMLayerState(p_layer->output_dim());
	_dc_next = Tensor::Zero({ p_layer->output_dim() });
	_dh_next = Tensor::Zero({ p_layer->output_dim() });
}


LSTMLayerGradient::~LSTMLayerGradient()
{
	delete _state;
}

void LSTMLayerGradient::set_delta(Tensor* p_delta)
{
	LSTMLayer* layer = dynamic_cast<LSTMLayer*>(_layer);
	Tensor zero = Tensor::Zero({ layer->_output_group->get_dim() });
	Tensor delta = Tensor::Concat(*p_delta, zero);
	_delta[layer->_output_group->get_id()].override(&delta);
}

void LSTMLayerGradient::init()
{
	LSTMLayer* layer = dynamic_cast<LSTMLayer*>(_layer);
	_deriv[layer->_output_group->get_id()] = Tensor::Zero({ layer->_output_group->get_dim() });
	_delta[layer->_output_group->get_id()] = Tensor::Zero({ 2 * layer->_output_group->get_dim() });
	_deriv["_h" + _layer->id()] = Tensor::Zero({ layer->_output_group->get_dim() });
	_delta["_h" + _layer->id()] = Tensor::Zero({ layer->_output_group->get_dim() });
	_deriv["_c" + _layer->id()] = Tensor::Zero({ layer->_output_group->get_dim() });
	_delta["_c" + _layer->id()] = Tensor::Zero({ layer->_output_group->get_dim() });
	_deriv[layer->_hf->get_id()] = Tensor::Zero({ layer->_hf->get_dim() });
	_delta[layer->_hf->get_id()] = Tensor::Zero({ layer->_hf->get_dim() });
	_deriv[layer->_hc->get_id()] = Tensor::Zero({ layer->_hc->get_dim() });
	_delta[layer->_hc->get_id()] = Tensor::Zero({ layer->_hc->get_dim() });
	_deriv[layer->_hi->get_id()] = Tensor::Zero({ layer->_hi->get_dim() });
	_delta[layer->_hi->get_id()] = Tensor::Zero({ layer->_hi->get_dim() });
	_deriv[layer->_ho->get_id()] = Tensor::Zero({ layer->_ho->get_dim() });
	_delta[layer->_ho->get_id()] = Tensor::Zero({ layer->_ho->get_dim() });
}

void LSTMLayerGradient::calc_deriv()
{
	LSTMLayer* layer = dynamic_cast<LSTMLayer*>(_layer);
	calc_deriv_group(layer->_output_group);
	calc_deriv_group(layer->_hf);
	calc_deriv_group(layer->_hc);
	calc_deriv_group(layer->_hi);
	calc_deriv_group(layer->_ho);
}

void LSTMLayerGradient::calc_delta(Tensor* p_weights, LayerState* p_state)
{
	LSTMLayer* layer = dynamic_cast<LSTMLayer*>(_layer);

	LSTMLayerState* state = dynamic_cast<LSTMLayerState*> (p_state);

	if (state != nullptr)
	{
		_dc_next.override(&state->dc_next);
		_dh_next.override(&state->dh_next);
	}
	else
	{
		_dc_next.fill(0);
		_dh_next.fill(0);
	}

	Tensor wd = p_weights->T() * p_state->delta;
	_delta[layer->_output_group->get_id()] = Tensor::apply(wd, _deriv[layer->_output_group->get_id()], Tensor::ew_dot);
	_delta["_h" + _layer->id()] = _delta[layer->_output_group->get_id()] * layer->_Wy->get_weights()->T() + _dh_next;

	for(int i = 0; i < layer->_output_group->get_dim(); i++)
	{
		_delta["_c" + _layer->id()][i] = layer->_ho->get_output()->at(i) * _delta["_h" + _layer->id()][i] * ActivationFunctionsDeriv::dtanh(layer->_c->at(i)) + _dc_next[i];
		_delta[layer->_ho->get_id()][i] = ActivationFunctionsDeriv::dsigmoid(layer->_ho->get_output()->at(i)) * tanh(layer->_c->at(i)) * _delta["_h" + _layer->id()][i];
		_delta[layer->_hf->get_id()][i] = ActivationFunctionsDeriv::dsigmoid(layer->_hf->get_output()->at(i)) * layer->_c_old->at(i) * _delta["_c" + _layer->id()][i];
		_delta[layer->_hi->get_id()][i] = ActivationFunctionsDeriv::dsigmoid(layer->_hi->get_output()->at(i)) * layer->_hc->get_output()->at(i) * _delta["_c" + _layer->id()][i];
		_delta[layer->_hc->get_id()][i] = ActivationFunctionsDeriv::dtanh(layer->_hc->get_output()->at(i)) * layer->_hi->get_output()->at(i) * _delta["_c" + _layer->id()][i];
	}

	Tensor dX = _delta[layer->_hf->get_id()] * layer->_Wf->get_weights()->T() + _delta[layer->_hi->get_id()] * layer->_Wi->get_weights()->T() + _delta[layer->_ho->get_id()] * layer->_Wo->get_weights()->T() + _delta[layer->_hc->get_id()] * layer->_Wc->get_weights()->T();
	for(int i = 0; i < layer->_x->size(); i++)
	{
		_dh_next[i] = dX[i];
	}

	_dc_next = Tensor::apply(*layer->_hf->get_output(), _delta["_c" + _layer->id()], Tensor::ew_dot);
}

void LSTMLayerGradient::update(map<string, Tensor>& p_update)
{
	IGradientComponent::update(p_update);

	LSTMLayer* layer = dynamic_cast<LSTMLayer*>(_layer);
	layer->_output_group->update_bias(p_update[layer->_output_group->get_id()]);
	layer->_hf->update_bias(p_update[layer->_hf->get_id()]);
	layer->_hi->update_bias(p_update[layer->_hi->get_id()]);
	layer->_ho->update_bias(p_update[layer->_ho->get_id()]);
	layer->_hc->update_bias(p_update[layer->_hc->get_id()]);
}

void LSTMLayerGradient::calc_gradient(map<string, Tensor>& p_w_gradient, map<string, Tensor>& p_b_gradient)
{
	LSTMLayer* layer = dynamic_cast<LSTMLayer*>(_layer);

	p_w_gradient[layer->_Wy->get_id()] = *layer->_h  * _delta[layer->_output_group->get_id()];
	p_b_gradient[layer->_output_group->get_id()] = _delta[layer->_output_group->get_id()];
	p_w_gradient[layer->_Wf->get_id()] = *layer->_x  * _delta[layer->_hf->get_id()];
	p_b_gradient[layer->_hf->get_id()] = _delta[layer->_hf->get_id()];
	p_w_gradient[layer->_Wi->get_id()] = *layer->_x  * _delta[layer->_hi->get_id()];
	p_b_gradient[layer->_hi->get_id()] = _delta[layer->_hi->get_id()];
	p_w_gradient[layer->_Wo->get_id()] = *layer->_x  * _delta[layer->_ho->get_id()];
	p_b_gradient[layer->_ho->get_id()] = _delta[layer->_ho->get_id()];
	p_w_gradient[layer->_Wc->get_id()] = *layer->_x  * _delta[layer->_hc->get_id()];
	p_b_gradient[layer->_hc->get_id()] = _delta[layer->_hc->get_id()];
}
