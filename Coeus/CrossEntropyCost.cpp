#include "CrossEntropyCost.h"
#include <cmath>

using namespace Coeus;

CrossEntropyCost::CrossEntropyCost()
{
}


CrossEntropyCost::~CrossEntropyCost()
{
}

float CrossEntropyCost::cost(Tensor * p_prediction, Tensor * p_target)
{
	float r = 0;

	for (int i = 0; i < p_prediction->size(); i++) {
		float e = p_target->at(i);
		float a = p_prediction->at(i);

		r += e * log(a);
	}

	return -r;
}

Tensor CrossEntropyCost::cost_deriv(Tensor * p_prediction, Tensor * p_target)
{
	float* data = Tensor::alloc_arr(p_prediction->size());
	int* shape = Tensor::copy_shape(p_prediction->rank(), p_prediction->shape());

	for (int i = 0; i < p_prediction->size(); i++) {
		float e = p_target->at(i);
		float a = p_prediction->at(i);

		data[i] = (a - e);
	}

	return Tensor(p_prediction->rank(), shape, data);
}
