#pragma once
#include "IInterpolation.h"

namespace Coeus
{
	class __declspec(dllexport) LinearInterpolation : public IInterpolation
	{
	public:
		LinearInterpolation(float p_point1, float p_point2, int p_interval);
		~LinearInterpolation();

		float interpolate(int p_t) override;
	};
}
