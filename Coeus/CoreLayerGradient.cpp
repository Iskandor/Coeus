#include "CoreLayerGradient.h"

using namespace Coeus;

CoreLayerGradient::CoreLayerGradient(CoreLayer* p_layer) : IGradientComponent(p_layer) {
}

CoreLayerGradient::~CoreLayerGradient()
{
}

void CoreLayerGradient::init()
{
	BaseCellGroup* g = get_layer<CoreLayer>()->_output_group;
	_deriv[g->get_id()] = Tensor::Zero({ g->get_dim() });
	_delta[g->get_id()] = Tensor::Zero({ g->get_dim() });
}

void CoreLayerGradient::calc_deriv() {
	calc_deriv_group(get_layer<CoreLayer>()->_output_group);
}

void CoreLayerGradient::calc_delta(Tensor* p_weights, Tensor* p_delta) {
	BaseCellGroup* g = get_layer<CoreLayer>()->_output_group;
	_delta[g->get_id()] = _deriv[g->get_id()] * (p_weights->T() * *p_delta);
}

void CoreLayerGradient::calc_gradient(map<string, Tensor> &p_w_gradient, map<string, Tensor> &p_b_gradient) {
	BaseCellGroup* g = get_layer<CoreLayer>()->_output_group;
	p_b_gradient[g->get_id()] = _delta[g->get_id()];
}
