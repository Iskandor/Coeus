#pragma once
#include <cmath>
#include "ILearningRateModule.h"

namespace Coeus {
	class __declspec(dllexport) AnnealingScheduler : public ILearningRateModule
	{
	public:
		AnnealingScheduler(int p_model_size);
		~AnnealingScheduler();

		double get_alpha();

	private:
		const double _warmup = pow(400, -1.5);
		int _model_size;
		int _step;
		
	};
}
