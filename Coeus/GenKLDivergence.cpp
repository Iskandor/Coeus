#include "GenKLDivergence.h"
#include <cmath>

using namespace Coeus;

GenKLDivergence::GenKLDivergence()
{
}


GenKLDivergence::~GenKLDivergence()
{
}

double GenKLDivergence::cost(Tensor * p_prediction, Tensor * p_target)
{
	double r = 0;
	double sum1 = 0;
	double sum2 = 0;
	double sum3 = 0;

	for (int i = 0; i < p_prediction->size(); i++) {
		double e = p_target->at(i);
		double a = p_prediction->at(i);

		sum1 += e * log(e / a);
		sum2 += e;
		sum3 += a;
	}

	r = sum1 - sum2 + sum3;

	return r;
}

Tensor Coeus::GenKLDivergence::cost_deriv(Tensor * p_prediction, Tensor * p_target)
{
	double* data = Tensor::alloc_arr(p_prediction->size());
	int* shape = Tensor::copy_shape(p_prediction->rank(), p_prediction->shape());

	for (int i = 0; i < p_prediction->size(); i++) {
		double e = p_target->at(i);
		double a = p_prediction->at(i);

		data[i] = (e + a) / a;
	}

	return Tensor(p_prediction->rank(), shape, data);
}
