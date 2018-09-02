#include "GaussianActivation.h"
#include <cmath>
#include "FLAB.h"
#include <cassert>

using namespace Coeus;

GaussianActivation::GaussianActivation(const double p_sigma): _sigma(p_sigma) {
}

GaussianActivation::~GaussianActivation()
{
}

Tensor GaussianActivation::activate(Tensor& p_input) {
	double* arr = Tensor::alloc_arr(p_input.size());

	for (int i = 0; i < p_input.size(); i++) {
		arr[i] = 1.0 / sqrt(2 * PI * pow(_sigma, 2)) * exp(-(pow(p_input[i], 2) / 2 * pow(_sigma, 2)));
	}

	return Tensor({ p_input.size() }, arr);
}

Tensor GaussianActivation::deriv(Tensor& p_input) {
	assert(0);
	return Tensor::Zero({ p_input.size() }).diag();
}
