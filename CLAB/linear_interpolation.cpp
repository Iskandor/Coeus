#include "linear_interpolation.h"

linear_interpolation::linear_interpolation(const float p_point1, const float p_point2, const int p_interval) : iinterpolation(p_point1, p_point2, p_interval)
{
}

linear_interpolation::~linear_interpolation()
= default;

float linear_interpolation::interpolate(const int p_t)
{
	float t = 1;

	if (p_t <= _interval) {
		t = static_cast<float>(p_t) / _interval;
	}

	return (1 - t) * _point1 + t * _point2;
}
