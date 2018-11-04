#pragma once
#include "ILearningRateModule.h"

namespace Coeus
{
	class __declspec(dllexport) WarmStartup : public ILearningRateModule
	{
	public:
		WarmStartup(double p_alpha_min, double p_alpha_max, int p_T0, int p_Tmult);
		~WarmStartup();

		double get_alpha() override;

	private:
		double _alpha_min;
		double _alpha_max;
		int _Tcur;
		int _Ti;
		int _Tmult;
	};
}
