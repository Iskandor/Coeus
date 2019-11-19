#include "LinearInterpolation.h"

using namespace Coeus;

LinearInterpolation::LinearInterpolation(const float p_point1, const float p_point2, const int p_interval) : IInterpolation(p_point1, p_point2, p_interval)
{
}

LinearInterpolation::~LinearInterpolation()
= default;

float LinearInterpolation::interpolate(const int p_t)
{
	float t = 1;

	if (p_t <= _interval) {
		t = static_cast<float>(p_t) / _interval;
	}

	return (1 - t) * _point1 + t * _point2;
}
