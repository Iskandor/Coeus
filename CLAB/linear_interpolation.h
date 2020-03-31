#pragma once
#include "iinterpolation.h"

class __declspec(dllexport) linear_interpolation : public iinterpolation
{
public:
	linear_interpolation(float p_point1, float p_point2, int p_interval);
	~linear_interpolation();

	float interpolate(int p_t) override;
};