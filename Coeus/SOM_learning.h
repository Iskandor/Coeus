#pragma once
#include "SOM.h"
#include "SOM_params.h"
#include "Base_SOM_learning.h"

namespace Coeus
{
	class __declspec(dllexport) SOM_learning : public Base_SOM_learning
	{
	public:
		explicit SOM_learning(SOM* p_som, SOM_params* p_params, SOM_analyzer* p_analyzer);
		virtual ~SOM_learning();

		void train(Tensor *p_input) override;

	private:
		Tensor	_delta_w;

		SOM* _som;
	};

}


