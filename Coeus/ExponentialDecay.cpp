#include "ExponentialDecay.h"
#include <cmath>
#include <iostream>

using namespace Coeus;
ExponentialDecay::ExponentialDecay(const double p_alpha0, const double p_k): 
	_alpha0(p_alpha0), 
	_k(p_k), 
	_t(0)
{
}

ExponentialDecay::~ExponentialDecay()
= default;

double ExponentialDecay::get_alpha()
{
	const double alpha = _alpha0 * exp(-_k*_t);
	_t++;

	return alpha;
}
