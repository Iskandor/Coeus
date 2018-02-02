#pragma once
#include "Base_SOM_params.h"

namespace Coeus
{
	class __declspec(dllexport) SOM_params : public Base_SOM_params
	{
	public:
		SOM_params(SOM* p_som);
		~SOM_params();

		void init_training(double p_alpha, double p_epochs);
		void param_decay() override;

		double alpha() const { return _alpha; }

	private:
		double _alpha0;
		double _alpha;
	};
}


