#include "exponential_interpolation.h"
#include <cmath>


exponential_interpolation::exponential_interpolation(const float p_point1, const float p_point2, const int p_interval) : iinterpolation(p_point1, p_point2, p_interval)
{
	if (_point1 == 0) _point1 += 1e-8f;
	if (_point2 == 0) _point2 += 1e-8f;
}

exponential_interpolation::~exponential_interpolation()
= default;

float exponential_interpolation::interpolate(const int p_t)
{
	float t = 1;

	if (p_t <= _interval) {
		t = static_cast<float>(p_t) / static_cast<float>(_interval);
	}

	return _point1 * pow(_point2 / _point1, t);
}

