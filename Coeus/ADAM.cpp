#include "ADAM.h"
#include "ADAMRule.h"

using namespace Coeus;


ADAM::ADAM(NeuralNetwork* p_network) : GradientAlgorithm(p_network), 
	_t(0)
{
}

ADAM::~ADAM() = default;

void ADAM::init(ICostFunction* p_cost_function, const double p_alpha, const double p_beta1, const double p_beta2, const double p_epsilon) {
	GradientAlgorithm::init(p_cost_function, new ADAMRule(_network_gradient, p_alpha, p_beta1, p_beta2, p_epsilon));
	_t = 0;
}

double ADAM::train(Tensor* p_input, Tensor* p_target)
{
	_t++;
	dynamic_cast<ADAMRule*>(_update_rule)->set_step(_t);
	return GradientAlgorithm::train(p_input, p_target);
}

double ADAM::train(vector<Tensor*>* p_input, vector<Tensor*>* p_target, const int p_batch)
{
	_t++;
	dynamic_cast<ADAMRule*>(_update_rule)->set_step(_t);
	return GradientAlgorithm::train(p_input, p_target, p_batch);
}
