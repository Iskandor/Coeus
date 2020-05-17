#include "linear_interpolation.h"

/**
* \brief Linear interpolation class
* \param p_point1 initial value in time t0
* \param p_point2 final value in time tk
* \param p_interval number of timesteps k
 */
linear_interpolation::linear_interpolation(const float p_point1, const float p_point2, const int p_interval) : iinterpolation(p_point1, p_point2, p_interval)
{
}

linear_interpolation::~linear_interpolation()
= default;

/**
* \brief Calculate value of point in time t from interval <0,k>
* \param p_t time t
* \return value for time t
*/
float linear_interpolation::interpolate(const int p_t)
{
	float t = 1;

	if (p_t <= _interval) {
		t = static_cast<float>(p_t) / _interval;
	}

	return (1 - t) * _point1 + t * _point2;
}
