#include "ExponentialCost.h"
#include <cmath>

using namespace Coeus;

ExponentialCost::ExponentialCost(double p_tau)
{
	_tau = p_tau;
}


ExponentialCost::~ExponentialCost()
{
}

double ExponentialCost::cost(Tensor * p_prediction, Tensor * p_target)
{
	double r = 0;

	for (int i = 0; i < p_prediction->size(); i++) {
		double e = p_target->at(i);
		double a = p_prediction->at(i);

		r += pow(a - e, 2);
	}

	r = _tau * exp(1 / _tau * r);

	return r;
}

Tensor ExponentialCost::cost_deriv(Tensor * p_prediction, Tensor * p_target)
{
	double* data = Tensor::alloc_arr(p_prediction->size());
	int* shape = Tensor::copy_shape(p_prediction->rank(), p_prediction->shape());
	double c = cost(p_prediction, p_target);


	for (int i = 0; i < p_prediction->size(); i++) {
		double e = p_target->at(i);
		double a = p_prediction->at(i);

		data[i] = 2 / _tau * (a - e) * c;
	}

	return Tensor(p_prediction->rank(), shape, data);
}
