#include "CrossEntropyCost.h"
#include <cmath>

using namespace Coeus;

CrossEntropyCost::CrossEntropyCost()
{
	_n = 0;
	_mean_cost = 0;
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

		if (a == 1) a -= 1e-6f;
		if (a == 0) a += 1e-6f;
		
		r += e * log(a) + (1 - e) * log(1 - a);
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

		if (a == 1) a -= 1e-6f;
		if (a == 0) a += 1e-6f;

		data[i] = -(e / a - (1 - e) / (1 - a));
	}

	return Tensor(p_prediction->rank(), shape, data);
}
