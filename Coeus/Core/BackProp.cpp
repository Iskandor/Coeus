#include "BackProph.h"
#include "BackPropRule.h"

using namespace Coeus;

BackProp::BackProp(NeuralNetwork* p_network) : GradientAlgorithm(p_network) {

}

BackProp::~BackProp() = default;

void BackProp::init(ICostFunction* p_cost_function, const float p_alpha, const float p_momentum, const bool p_nesterov) {
	GradientAlgorithm::init(p_cost_function, new BackPropRule(_network, p_alpha, p_momentum, p_nesterov));
}