#include "KLDivergence.h"
#include <cmath>

using namespace Coeus;

KLDivergence::KLDivergence()
{
}


KLDivergence::~KLDivergence()
{
}

float KLDivergence::cost(Tensor * p_prediction, Tensor * p_target)
{
	float r = 0;

	for (int i = 0; i < p_prediction->size(); i++) {
		float e = p_target->at(i);
		float a = p_prediction->at(i);

		r += e * log(e) - e * log(a);
	}

	return r;
}

Tensor KLDivergence::cost_deriv(Tensor * p_prediction, Tensor * p_target)
{
	float* data = Tensor::alloc_arr(p_prediction->size());
	int* shape = Tensor::copy_shape(p_prediction->rank(), p_prediction->shape());

	for (int i = 0; i < p_prediction->size(); i++) {
		float e = p_target->at(i);
		float a = p_prediction->at(i);

		data[i] = log(e) - log(a);
	}

	return Tensor(p_prediction->rank(), shape, data);
}

