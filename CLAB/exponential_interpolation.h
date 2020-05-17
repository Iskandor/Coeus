#pragma once
#include "iinterpolation.h"

class COEUS_DLL_API exponential_interpolation : public iinterpolation
{
public:
	exponential_interpolation(float p_point1, float p_point2, int p_interval);
	~exponential_interpolation();

	float interpolate(int p_t) override;
};
