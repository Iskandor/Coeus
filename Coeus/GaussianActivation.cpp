#include "GaussianActivation.h"
#include <cmath>
#include "FLAB.h"
#include <cassert>

using namespace Coeus;

GaussianActivation::GaussianActivation(const double p_sigma): IActivationFunction(GAUSS), _sigma(p_sigma) {
}

GaussianActivation::~GaussianActivation()
{
}

Tensor GaussianActivation::activate(Tensor& p_input) {
	double* arr = Tensor::alloc_arr(p_input.size());
	double* y = &arr[0];
	double* x = &p_input.arr()[0];

	for (int i = 0; i < p_input.size(); i++) {
		*y++ = 1.0 / sqrt(2 * PI * pow(_sigma, 2)) * exp(-(pow(*x++, 2) / 2 * pow(_sigma, 2)));
	}

	return Tensor({ p_input.size() }, arr);
}

Tensor GaussianActivation::derivative(Tensor& p_input) {
	assert(0);
	return Tensor::Zero({ p_input.size() }).diag();
}

json GaussianActivation::get_json()
{
	json result = IActivationFunction::get_json();

	result["sigma"] = _sigma;

	return result;
}
