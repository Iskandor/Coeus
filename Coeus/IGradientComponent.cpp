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

void IGradientComponent::calc_deriv_group(NeuralGroup* p_group) {
	switch (p_group->getActivationFunction()) {
	case NeuralGroup::BINARY:
		_deriv[p_group->get_id()] = p_group->getOutput()->apply(ActivationFunctionsDeriv::dbinary);
		break;
	case NeuralGroup::IDENTITY:
	case NeuralGroup::LINEAR:
		_deriv[p_group->get_id()] = p_group->getOutput()->apply(ActivationFunctionsDeriv::dlinear);
		break;
	case NeuralGroup::RELU:
		_deriv[p_group->get_id()] = p_group->getOutput()->apply(ActivationFunctionsDeriv::drelu);
		break;
	case NeuralGroup::SIGMOID:
		_deriv[p_group->get_id()] = p_group->getOutput()->apply(ActivationFunctionsDeriv::dsigmoid);
		break;
	case NeuralGroup::SOFTPLUS:
		_deriv[p_group->get_id()] = p_group->getOutput()->apply(ActivationFunctionsDeriv::dsoftplus);
		break;
	case NeuralGroup::TANH:
		_deriv[p_group->get_id()] = p_group->getOutput()->apply(ActivationFunctionsDeriv::dtanh);
		break;
	default:;
	}
}
