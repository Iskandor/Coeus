#pragma once
#include "IInterpolation.h"
namespace Coeus
{
	class __declspec(dllexport) ExponentialInterpolation : public IInterpolation
	{
	public:
		ExponentialInterpolation(float p_point1, float p_point2, int p_interval);
		~ExponentialInterpolation();

		float interpolate(int p_t) override;
	};
}