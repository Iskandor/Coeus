#include "GenKLDivergence.h"
#include <cmath>

using namespace Coeus;

GenKLDivergence::GenKLDivergence()
{
}


GenKLDivergence::~GenKLDivergence()
{
}

float GenKLDivergence::cost(Tensor * p_prediction, Tensor * p_target)
{
	float r = 0;
	float sum1 = 0;
	float sum2 = 0;
	float sum3 = 0;

	for (int i = 0; i < p_prediction->size(); i++) {
		float e = p_target->at(i);
		float a = p_prediction->at(i);

		sum1 += e * log(e / a);
		sum2 += e;
		sum3 += a;
	}

	r = sum1 - sum2 + sum3;

	return r;
}

Tensor Coeus::GenKLDivergence::cost_deriv(Tensor * p_prediction, Tensor * p_target)
{
	float* data = Tensor::alloc_arr(p_prediction->size());
	int* shape = Tensor::copy_shape(p_prediction->rank(), p_prediction->shape());

	for (int i = 0; i < p_prediction->size(); i++) {
		float e = p_target->at(i);
		float a = p_prediction->at(i);

		data[i] = (e + a) / a;
	}

	return Tensor(p_prediction->rank(), shape, data);
}
