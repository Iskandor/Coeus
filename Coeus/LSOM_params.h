#pragma once
#include "Base_SOM_params.h"
#include "LSOM.h"

namespace Coeus
{
	class __declspec(dllexport) LSOM_params : public Base_SOM_params
	{
	public:
		explicit LSOM_params(LSOM* p_lsom);
		~LSOM_params();

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

