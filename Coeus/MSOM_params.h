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

		void init_training(float p_gamma1, float p_gamma2, float p_epochs);
		void param_decay() override;

		float gamma1() const { return _gamma1; }
		float gamma2() const { return _gamma2; }

	private:
		float _gamma1_0;
		float _gamma1;
		float _gamma2_0;
		float _gamma2;
	};

}

