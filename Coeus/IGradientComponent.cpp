#include "IGradientComponent.h"
#include "ActivationFunctionsDeriv.h"

using namespace Coeus;

IGradientComponent::IGradientComponent(BaseLayer* p_layer)
{
	_state = nullptr;
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

	for (auto it = _layer->_groups.begin(); it != _layer->_groups.end(); ++it) {
		if (it->second->is_bias()) {
			it->second->update_bias(p_update[it->first]);
		}		
	}
}

LayerState* IGradientComponent::get_state()
{
	_state->delta.override(&_delta[_layer->_input_group->get_id()]);
	return _state;
}

void IGradientComponent::calc_deriv_group(NeuralGroup* p_group) {
	switch (p_group->get_activation_function()) {
	case BINARY:
		_deriv[p_group->get_id()] = Tensor::apply(*p_group->get_output(), ActivationFunctionsDeriv::dbinary);
		break;
	case LINEAR:
		_deriv[p_group->get_id()] = Tensor::apply(*p_group->get_output(), ActivationFunctionsDeriv::dlinear);
		break;
	case RELU:
		_deriv[p_group->get_id()] = Tensor::apply(*p_group->get_output(), ActivationFunctionsDeriv::drelu);
		break;
	case SIGMOID:
		_deriv[p_group->get_id()] = Tensor::apply(*p_group->get_output(), ActivationFunctionsDeriv::dsigmoid);
		break;
	case SOFTPLUS:
		_deriv[p_group->get_id()] = Tensor::apply(*p_group->get_output(), ActivationFunctionsDeriv::dsoftplus);
		break;
	case TANH:
		_deriv[p_group->get_id()] = Tensor::apply(*p_group->get_output(), ActivationFunctionsDeriv::dtanh);
		break;
	default:;
	}
}
