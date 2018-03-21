#pragma once
#include "Base_SOM_params.h"
#include "MSOM.h"

namespace Coeus
{

	class __declspec(dllexport) MSOM_params : public Base_SOM_params
	{
	public:
		explicit MSOM_params(MSOM* p_som);
		~MSOM_params();

		void init_training(double p_gamma1, double p_gamma2, double p_epochs);
		void param_decay() override;

		double gamma1() const { return _gamma1; }
		double gamma2() const { return _gamma2; }

	private:
		double _gamma1_0;
		double _gamma1;
		double _gamma2_0;
		double _gamma2;
	};

}

