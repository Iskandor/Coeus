#include "CrossEntropyCost.h"
#include <cmath>

using namespace Coeus;

CrossEntropyCost::CrossEntropyCost()
{
}


CrossEntropyCost::~CrossEntropyCost()
{
}

double CrossEntropyCost::cost(Tensor * p_prediction, Tensor * p_target)
{
	double r = 0;

	for (int i = 0; i < p_prediction->size(); i++) {
		double e = p_target->at(i);
		double a = p_prediction->at(i);

		r += e * log(a) + (1 - e) * log(1 - a);
	}

	return -r;
}

Tensor CrossEntropyCost::cost_deriv(Tensor * p_prediction, Tensor * p_target)
{
	double* data = Tensor::alloc_arr(p_prediction->size());

	for (int i = 0; i < p_prediction->size(); i++) {
		double e = p_target->at(i);
		double a = p_prediction->at(i);

		data[i] = (a - e) / ((1 - a) * a);
	}

	return Tensor(p_prediction->rank(), p_prediction->shape(), data);
}