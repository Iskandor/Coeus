#include "NaturalGradient.h"
#include "TensorOperator.h"
#include <omp.h>

using namespace Coeus;

int NaturalGradient::_n = 0;

NaturalGradient::NaturalGradient(NeuralNetwork* p_network) : NetworkGradient(p_network)
{
}


NaturalGradient::~NaturalGradient()
= default;

void NaturalGradient::calc_gradient(Tensor* p_loss) {

	NetworkGradient::calc_gradient(p_loss);
	calc_hessian(_gradient);

	_gradient._gradient = _gradient._hessian.inv() * _gradient._gradient;

	_gradient.reshape();
}

void NaturalGradient::calc_hessian(Gradient& p_gradient)
{
	p_gradient.flatten();
	_n++;
	p_gradient._hessian += (p_gradient._hessian - p_gradient._gradient * p_gradient._gradient.T()) * (1.f / _n);
}