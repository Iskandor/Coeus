#include "QuadraticCost.h"
#include <cmath>

using namespace Coeus;

QuadraticCost::QuadraticCost()
{
}


QuadraticCost::~QuadraticCost()
{
}

double Coeus::QuadraticCost::cost(Tensor * p_prediction, Tensor * p_target)
{
	double r = 0;

	for (int i = 0; i < p_prediction->size(); i++) {
		double e = p_target->at(i);
		double a = p_prediction->at(i);
		r += pow(a - e, 2);
	}

	r *= 0.5;

	return r;
}

Tensor Coeus::QuadraticCost::cost_deriv(Tensor * p_prediction, Tensor * p_target)
{
	double* data = Tensor::alloc_arr(p_prediction->size());

	for (int i = 0; i < p_prediction->size(); i++) {
		double e = p_target->at(i);
		double a = p_prediction->at(i);
		data[i] = a - e;
	}

	return Tensor(p_prediction->rank(), p_prediction->shape(), data);
}
