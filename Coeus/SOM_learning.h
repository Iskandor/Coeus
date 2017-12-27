#pragma once
#include "SOM.h"
#include "Base_SOM_learning.h"

namespace Coeus
{
	class __declspec(dllexport) SOM_learning : public Base_SOM_learning
	{
	public:
		explicit SOM_learning(SOM* p_som);
		virtual ~SOM_learning();

		void init_training(double p_alpha, double p_epochs);
		void train(Tensor *p_input) override;
		void param_decay() override;

	private:
		double _alpha0;
		double _alpha;

		SOM* _som;
	};

}


