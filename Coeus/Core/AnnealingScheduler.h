#pragma once
#include <cmath>
#include "ILearningRateModule.h"

namespace Coeus {
	class __declspec(dllexport) AnnealingScheduler : public ILearningRateModule
	{
	public:
		AnnealingScheduler(int p_model_size);
		~AnnealingScheduler();

		float get_alpha();

	private:
		const float _warmup = pow(400.f, -1.5f);
		int _model_size;
		int _step;
		
	};
}
