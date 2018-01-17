#pragma once

#include <Tensor.h>
#include "MSOM.h"
#include "Base_SOM_learning.h"

using namespace FLAB;

namespace Coeus
{
	class __declspec(dllexport) MSOM_learning : public Base_SOM_learning
	{
	public:
		explicit MSOM_learning(MSOM *p_msom);
		~MSOM_learning();

		void init_training(double p_gamma1, double p_gamma2, double p_epochs);

		void train(Tensor *p_input) override;
		void param_decay() override;

	private:
		double _gamma1_0;
		double _gamma1;
		double _gamma2_0;
		double _gamma2;

		Tensor	_delta_c;

		MSOM* _msom;

		
	};
}


