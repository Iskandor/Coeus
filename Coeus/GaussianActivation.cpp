#include "GaussianActivation.h"
#include <cmath>
#include <cassert>

using namespace Coeus;

GaussianActivation::GaussianActivation(const float p_sigma): IActivationFunction(GAUSS), _sigma(p_sigma) {
}

GaussianActivation::~GaussianActivation()
{
}

Tensor* GaussianActivation::backward(Tensor* p_input)
{
	return nullptr;
}

Tensor* GaussianActivation::forward(Tensor* p_input)
{
	IActivationFunction::forward(p_input);
	float* y = &_output->arr()[0];
	float* x = &_input->arr()[0];

	for (int i = 0; i < _output->size(); i++) {
		*y++ = 1.0 / sqrt(2 * PI * pow(_sigma, 2)) * exp(-(pow(*x++, 2) / 2 * pow(_sigma, 2)));
	}

	return _output;
}

Tensor GaussianActivation::derivative(Tensor& p_input) {
	assert(0);
	return Tensor::Zero({ p_input.size() });
}

json GaussianActivation::get_json()
{
	json result = IActivationFunction::get_json();

	result["sigma"] = _sigma;

	return result;
}
