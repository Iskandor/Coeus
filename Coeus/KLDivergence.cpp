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
	float entropy = 0;
	float cross_entropy = 0;

	for (int i = 0; i < p_prediction->size(); i++) {
		float p = p_target->at(i);
		float q = p_prediction->at(i);

		entropy += p * log(p);
		cross_entropy += p * log(q);
	}

	return entropy - cross_entropy;
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

