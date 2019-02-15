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

		void init_training(float p_alpha, float p_beta, float p_epochs);
		void param_decay() override;

		float alpha() const { return _alpha; }
		float beta() const { return _beta; }

	private:
		float _alpha0;
		float _alpha;
		float _beta0;
		float _beta;
	};
}

