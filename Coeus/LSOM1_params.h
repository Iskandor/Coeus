#pragma once
#include "Base_SOM_params.h"
#include "LSOM1.h"

namespace Coeus
{
	class __declspec(dllexport) LSOM1_params : public Base_SOM_params
	{
	public:
		explicit LSOM1_params(LSOM1* p_lsom);
		~LSOM1_params();

		void init_training(double p_alpha, double p_beta, double p_epochs);
		void param_decay() override;

		double alpha() const { return _alpha; }
		double beta() const { return _beta; }

	private:
		double _alpha0;
		double _alpha;
		double _beta0;
		double _beta;
	};
}

