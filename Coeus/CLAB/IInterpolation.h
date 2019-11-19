#pragma once
namespace Coeus
{
	class __declspec(dllexport) IInterpolation
	{
	public:
		IInterpolation(float p_point1, float p_point2, int p_interval);
		virtual ~IInterpolation();

		virtual float interpolate(int p_t) = 0;

	protected:
		float _point1;
		float _point2;
		int _interval;
	};

	inline IInterpolation::IInterpolation(const float p_point1, const float p_point2, const int p_interval): _point1(p_point1), _point2(p_point2), _interval(p_interval)
	{
	}

	inline IInterpolation::~IInterpolation()
	= default;
}
