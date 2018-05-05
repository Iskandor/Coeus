#include "IGradientComponent.h"
#include "ActivationFunctionsDeriv.h"

using namespace Coeus;

IGradientComponent::IGradientComponent(BaseLayer* p_layer)
{
	_layer = p_layer;
}


IGradientComponent::~IGradientComponent()
{
}

void IGradientComponent::set_delta(Tensor* p_delta) {
	_delta[_layer->_output_group->get_id()].override(p_delta);
}

void IGradientComponent::calc_gradient(map<string, Tensor> &p_w_gradient, map<string, Tensor> &p_b_gradient) {
}

void IGradientComponent::update(map<string, Tensor>& p_update) {
	for(auto it = _layer->_connections.begin(); it != _layer->_connections.end(); ++it) {
		it->second->update_weights(p_update[it->first]);
	}
}

void IGradientComponent::calc_deriv_group(NeuralGroup* p_group) {
	switch (p_group->get_activation_function()) {
	case NeuralGroup::BINARY:
		_deriv[p_group->get_id()] = Tensor::apply(p_group->get_output(), ActivationFunctionsDeriv::dbinary);
		break;
	case NeuralGroup::IDENTITY:
	case NeuralGroup::LINEAR:
		_deriv[p_group->get_id()] = Tensor::apply(p_group->get_output(), ActivationFunctionsDeriv::dlinear);
		break;
	case NeuralGroup::RELU:
		_deriv[p_group->get_id()] = Tensor::apply(p_group->get_output(), ActivationFunctionsDeriv::drelu);
		break;
	case NeuralGroup::SIGMOID:
		_deriv[p_group->get_id()] = Tensor::apply(p_group->get_output(), ActivationFunctionsDeriv::dsigmoid);
		break;
	case NeuralGroup::SOFTPLUS:
		_deriv[p_group->get_id()] = Tensor::apply(p_group->get_output(), ActivationFunctionsDeriv::dsoftplus);
		break;
	case NeuralGroup::TANH:
		_deriv[p_group->get_id()] = Tensor::apply(p_group->get_output(), ActivationFunctionsDeriv::dtanh);
		break;
	default:;
	}
}
