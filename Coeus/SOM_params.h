#pragma once
#include "Base_SOM_params.h"

namespace Coeus
{
	class __declspec(dllexport) SOM_params : public Base_SOM_params
	{
	public:
		SOM_params(SOM* p_som);
		~SOM_params();

		void init_training(float p_alpha, float p_epochs);
		void param_decay() override;

		float alpha() const { return _alpha; }

	private:
		float _alpha0;
		float _alpha;
	};
}


