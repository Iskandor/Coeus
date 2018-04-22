#include "CoreLayerGradient.h"

using namespace Coeus;

CoreLayerGradient::CoreLayerGradient(CoreLayer* p_layer) : IGradientComponent(p_layer) {
}

CoreLayerGradient::~CoreLayerGradient()
{
}

void Coeus::CoreLayerGradient::init()
{
	NeuralGroup* g = reinterpret_cast<CoreLayer*>(_layer)->_output_group;
	_deriv[g->get_id()] = Tensor::Zero({ g->get_dim() });
	_delta[g->get_id()] = Tensor::Zero({ g->get_dim() });
}

void CoreLayerGradient::calc_deriv() {
	calc_deriv_group(reinterpret_cast<CoreLayer*>(_layer)->_output_group);
}

void CoreLayerGradient::calc_delta(Tensor* p_weights, Tensor* p_delta) {
	NeuralGroup* g = reinterpret_cast<CoreLayer*>(_layer)->_output_group;
	_delta[g->get_id()] = p_weights->T() * *p_delta;
}

void CoreLayerGradient::calc_gradient(map<string, Tensor>& p_gradient) {
}
