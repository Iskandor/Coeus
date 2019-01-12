#include "PowerSign.h"
#include "PowerSignRule.h"

using namespace Coeus;

PowerSign::PowerSign(NeuralNetwork* p_network) : BaseGradientAlgorithm(p_network)
{
}


PowerSign::~PowerSign()
{
}

void PowerSign::init(ICostFunction* p_cost_function, const double p_alpha)
{
	BaseGradientAlgorithm::init(p_cost_function, new PowerSignRule(_network_gradient, p_alpha));
}
