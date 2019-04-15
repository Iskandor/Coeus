#include "ExponentialInterpolation.h"
#include <cmath>

using namespace Coeus;

ExponentialInterpolation::ExponentialInterpolation(const float p_point1, const float p_point2, const int p_interval) : IInterpolation(p_point1, p_point2, p_interval)
{
	if (_point1 == 0) _point1 += 1e-8;
	if (_point2 == 0) _point2 += 1e-8;
}

ExponentialInterpolation::~ExponentialInterpolation()
= default;

float ExponentialInterpolation::interpolate(const int p_t)
{
	const float t = static_cast<float>(p_t) / _interval;
	return _point1 * pow(_point2 / _point1, t);
}

