#include "ActivationFunctions.h"
#include <cmath>
#include <algorithm>
#include "FLAB.h"

using namespace Coeus;

double Coeus::ActivationFunctions::linear(double p_x)
{
	return p_x;
}

double Coeus::ActivationFunctions::binary(double p_x)
{
	return p_x > 0 ? 1 : 0;
}

double Coeus::ActivationFunctions::sigmoid(double p_x)
{
	return 1 / (1 + exp(-p_x));
}

double ActivationFunctions::tanh(double p_x) {
	return std::tanh(p_x);
}

double ActivationFunctions::exponential(double p_x) {
	return exp(-p_x);
}

double ActivationFunctions::softplus(double p_x) {
	return log(1 + exp(p_x));
}

double ActivationFunctions::relu(double p_x) {
	return std::max(0., p_x);
}

double ActivationFunctions::kexponential(double p_x) {
	const int k = 5;
	return exp(-k * p_x);
}

double ActivationFunctions::gauss(double p_x) {
	const double sigma = 0.2;
	return 1.0 / sqrt(2 * PI * pow(sigma, 2)) * exp(-(pow(p_x, 2) / 2 * pow(sigma, 2)));;
}
