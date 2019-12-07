#include "PowerSign.h"
#include "PowerSignRule.h"

using namespace Coeus;

PowerSign::PowerSign(NeuralNetwork* p_network) : GradientAlgorithm(p_network)
{
}


PowerSign::~PowerSign()
{
}

void PowerSign::init(ICostFunction* p_cost_function, const float p_alpha)
{
	GradientAlgorithm::init(p_cost_function, new PowerSignRule(_network, p_alpha));
}
