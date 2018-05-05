#pragma once
#include "Base_SOM_learning.h"
#include "LISSOM.h"
#include "LISSOM_params.h"

namespace Coeus
{

	class __declspec(dllexport) LISSOM_learning : public Base_SOM_learning
	{
	public:
		LISSOM_learning(LISSOM* p_som, LISSOM_params* p_params, SOM_analyzer* p_som_analyzer);
		~LISSOM_learning();

		void train(Tensor *p_input) override;

	private:
		LISSOM* _lissom;

		Tensor	_delta_aw;
		Tensor	_delta_ew;
		Tensor	_delta_iw;
	};

}


