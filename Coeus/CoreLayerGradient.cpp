#include "CoreLayerGradient.h"

using namespace Coeus;

CoreLayerGradient::CoreLayerGradient(CoreLayer* p_layer) : IGradientComponent(p_layer) {
	_state = new LayerState(p_layer->get_output_group()->get_dim());
}

CoreLayerGradient::~CoreLayerGradient()
{
	delete _state;
}

void CoreLayerGradient::init()
{
	NeuralGroup* g = reinterpret_cast<CoreLayer*>(_layer)->_output_group;
	_deriv[g->get_id()] = Tensor::Zero({ g->get_dim() });
	_delta[g->get_id()] = Tensor::Zero({ g->get_dim() });
}

void CoreLayerGradient::calc_deriv() {
	calc_deriv_group(reinterpret_cast<CoreLayer*>(_layer)->_output_group);
}

void CoreLayerGradient::calc_delta(Tensor* p_weights, LayerState* p_state) {
	NeuralGroup* g = reinterpret_cast<CoreLayer*>(_layer)->_output_group;
	Tensor wd = p_weights->T() * p_state->delta;
	_delta[g->get_id()] = Tensor::apply(wd, _deriv[g->get_id()], Tensor::ew_dot);
}

void CoreLayerGradient::calc_gradient(map<string, Tensor> &p_w_gradient, map<string, Tensor> &p_b_gradient) {
	NeuralGroup* g = reinterpret_cast<CoreLayer*>(_layer)->_output_group;
	p_b_gradient[g->get_id()] = _delta[g->get_id()];
}

void CoreLayerGradient::update(map<string, Tensor>& p_update) {
	IGradientComponent::update(p_update);

	NeuralGroup* g = reinterpret_cast<CoreLayer*>(_layer)->_output_group;
	g->update_bias(p_update[g->get_id()]);
}
