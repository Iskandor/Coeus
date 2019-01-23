#include "BPTT.h"

using namespace Coeus;

BPTT::BPTT(NeuralNetwork* p_network, GradientAlgorithm* p_gradient_algorithm)
{
	_network = p_network;
	_gradient_algorithm = p_gradient_algorithm;
}

BPTT::~BPTT()
{
}

double BPTT::train(vector<Tensor*>* p_input, Tensor* p_target) const
{
	//return _gradient_algorithm->train(p_input, p_target);
	return 0;
}
