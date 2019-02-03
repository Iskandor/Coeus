#pragma once
#include "ILearningRateModule.h"

namespace Coeus {
	class __declspec(dllexport) ExponentialDecay : public ILearningRateModule
	{
	public:
		ExponentialDecay(double p_alpha0, double p_k);
		~ExponentialDecay();
		double get_alpha() override;

	private:
		double _alpha0;
		double _k;
		int	_t;
	};
}
