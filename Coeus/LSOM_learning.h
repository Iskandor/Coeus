#pragma once
#include "Base_SOM_learning.h"
#include "LSOM_params.h"
#include "LSOM.h"

namespace Coeus {

	class __declspec(dllexport) LSOM_learning : public Base_SOM_learning
	{
	public:
		LSOM_learning(LSOM* p_som, LSOM_params* p_params, SOM_analyzer* p_som_analyzer);
		~LSOM_learning();

		void train(Tensor *p_input) override;

		LSOM* lsom() const { return _lsom; }

	private:
		Tensor	_delta_w;
		Tensor	_delta_lw;

		LSOM*	_lsom;
	};
}