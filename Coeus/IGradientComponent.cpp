#include "IGradientComponent.h"

using namespace Coeus;

IGradientComponent::IGradientComponent(BaseLayer* p_layer, NeuralNetwork* p_network): _layer(p_layer),
                                                                                      _network(p_network) {
}

IGradientComponent::~IGradientComponent()
= default;

void IGradientComponent::set_delta(Tensor* p_delta) {
	_delta[_layer->_output_group->get_id()].override(p_delta);
}

void IGradientComponent::calc_gradient(map<string, Tensor> &p_gradient) {
}

void IGradientComponent::reset()
{
}

void IGradientComponent::calc_deriv_group(BaseCellGroup* p_group) {
	_deriv[p_group->get_id()] = p_group->get_deriv_output();
}
