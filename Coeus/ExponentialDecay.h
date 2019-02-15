#pragma once
#include "ILearningRateModule.h"

namespace Coeus {
	class __declspec(dllexport) ExponentialDecay : public ILearningRateModule
	{
	public:
		ExponentialDecay(float p_alpha0, float p_k);
		~ExponentialDecay();
		float get_alpha() override;

	private:
		float _alpha0;
		float _k;
		int	_t;
	};
}
