#include "ActivationFunctions.h"
#include <cmath>
#include <algorithm>
#include "FLAB.h"

using namespace Coeus;

float Coeus::ActivationFunctions::linear(float p_x)
{
	return p_x;
}

float Coeus::ActivationFunctions::binary(float p_x)
{
	return p_x > 0 ? 1.f : 0.f;
}

float Coeus::ActivationFunctions::sigmoid(float p_x)
{
	return 1 / (1 + exp(-p_x));
}

float ActivationFunctions::tanh(float p_x) {
	return std::tanh(p_x);
}

float ActivationFunctions::exponential(float p_x) {
	return exp(-p_x);
}

float ActivationFunctions::softplus(float p_x) {
	return log(1 + exp(p_x));
}

float ActivationFunctions::relu(float p_x) {
	return FLAB::max(0., p_x);
}

float ActivationFunctions::kexponential(float p_x) {
	const int k = 5;
	return exp(-k * p_x);
}

float ActivationFunctions::gauss(float p_x) {
	const float sigma = 0.2f;
	return 1.0f / sqrt(2 * FLAB::PI * pow(sigma, 2)) * exp(-(pow(p_x, 2) / 2 * pow(sigma, 2)));;
}
