#include "GaussianActivation.h"
#include <cmath>
#include <cassert>

using namespace Coeus;

GaussianActivation::GaussianActivation(const float p_sigma): IActivationFunction(GAUSS), _sigma(p_sigma) {
}

GaussianActivation::~GaussianActivation()
{
}

Tensor GaussianActivation::activate(Tensor& p_input) {
	float* arr = Tensor::alloc_arr(p_input.size());
	int* shape = Tensor::copy_shape(p_input.rank(), p_input.shape());
	float* y = &arr[0];
	float* x = &p_input.arr()[0];

	for (int i = 0; i < p_input.size(); i++) {
		*y++ = 1.0 / sqrt(2 * PI * pow(_sigma, 2)) * exp(-(pow(*x++, 2) / 2 * pow(_sigma, 2)));
	}

	return Tensor(p_input.rank(), shape, arr);
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
