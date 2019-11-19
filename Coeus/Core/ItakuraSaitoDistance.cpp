#include "ItakuraSaitoDistance.h"
#include <cmath>

using namespace Coeus;

ItakuraSaitoDistance::ItakuraSaitoDistance()
{
}


ItakuraSaitoDistance::~ItakuraSaitoDistance()
{
}

float ItakuraSaitoDistance::cost(Tensor * p_prediction, Tensor * p_target)
{
	float r = 0;

	for (int i = 0; i < p_prediction->size(); i++) {
		float e = p_target->at(i);
		float a = p_prediction->at(i);

		r += e / a - log(e / a) - 1 ;
	}

	return r;
}

Tensor ItakuraSaitoDistance::cost_deriv(Tensor * p_prediction, Tensor * p_target)
{
	float* data = Tensor::alloc_arr(p_prediction->size());
	int* shape = Tensor::copy_shape(p_prediction->rank(), p_prediction->shape());

	for (int i = 0; i < p_prediction->size(); i++) {
		float e = p_target->at(i);
		float a = p_prediction->at(i);

		data[i] = (e + pow(a, 2)) / pow(a,2);
	}

	return Tensor(p_prediction->rank(), shape, data);
}
