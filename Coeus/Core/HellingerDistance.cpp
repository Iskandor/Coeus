#include "HellingerDistance.h"
#include <cmath>

using namespace Coeus;

HellingerDistance::HellingerDistance()
{
}


HellingerDistance::~HellingerDistance()
{
}

float HellingerDistance::cost(Tensor * p_prediction, Tensor * p_target)
{
	float r = 0;

	for (int i = 0; i < p_prediction->size(); i++) {
		float e = p_target->at(i);
		float a = p_prediction->at(i);

		r += pow(sqrt(a) - sqrt(e), 2);
	}

	r *= 1 / sqrt(2);

	return r;
}

Tensor HellingerDistance::cost_deriv(Tensor * p_prediction, Tensor * p_target)
{
	float* data = Tensor::alloc_arr(p_prediction->size());
	int* shape = Tensor::copy_shape(p_prediction->rank(), p_prediction->shape());

	for (int i = 0; i < p_prediction->size(); i++) {
		float e = p_target->at(i);
		float a = p_prediction->at(i);

		data[i] = (sqrt(a) - sqrt(e))/(sqrt(2) * sqrt(a));
	}

	return Tensor(p_prediction->rank(), shape, data);
}
