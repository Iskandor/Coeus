#pragma once
#include "Tensor.h"
#include "IInterpolation.h"

namespace Coeus
{
	class __declspec(dllexport) BoltzmanExploration
	{
	public:
		BoltzmanExploration(float p_T, IInterpolation* p_interpolation = nullptr);
		~BoltzmanExploration();

		int get_action(Tensor* p_values);
		void update(int p_t);

	private:
		float _T;
		IInterpolation *_interpolation;
	};
}

