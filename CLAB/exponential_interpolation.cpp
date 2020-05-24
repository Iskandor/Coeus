#include "exponential_interpolation.h"
#include <cmath>


/**
 * \brief Exponential interpolation class
 * \param p_point1 initial value in time t0
 * \param p_point2 final value in time tk
 * \param p_interval number of timesteps k
 */
exponential_interpolation::exponential_interpolation(const float p_point1, const float p_point2, const int p_interval) : iinterpolation(p_point1, p_point2, p_interval)
{
	if (_point1 == 0) _point1 += 1e-8f;
	if (_point2 == 0) _point2 += 1e-8f;
}

exponential_interpolation::~exponential_interpolation()
= default;

/**
 * \brief Calculate value of point in time t from interval <0,k>
 * \param p_t time t
 * \return value for time t
 */
float exponential_interpolation::interpolate(const int p_t)
{
	float t = 1;

	if (p_t <= _interval) {
		t = static_cast<float>(p_t) / static_cast<float>(_interval);
	}

	return _point1 * pow(_point2 / _point1, t);
}
