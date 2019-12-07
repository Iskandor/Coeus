#pragma once
#include "ILearningRateModule.h"

namespace Coeus
{
	class __declspec(dllexport) WarmStartup : public ILearningRateModule
	{
	public:
		WarmStartup(float p_alpha_min, float p_alpha_max, int p_T0, int p_Tmult);
		~WarmStartup();

		float get_alpha() override;

	private:
		float _alpha_min;
		float _alpha_max;
		int _Tcur;
		int _Ti;
		int _Tmult;
	};
}
