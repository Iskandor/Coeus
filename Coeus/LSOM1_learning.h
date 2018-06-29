#pragma once
#include "Base_SOM_learning.h"
#include "LSOM1_params.h"
#include "LSOM1.h"

namespace Coeus {

	class __declspec(dllexport) LSOM1_learning : public Base_SOM_learning
	{
	public:
		LSOM1_learning(LSOM1* p_som, LSOM1_params* p_params, SOM_analyzer* p_som_analyzer);
		~LSOM1_learning();

		void train(Tensor *p_input) override;
		void update_friendship();

		LSOM1* lsom() const { return _lsom; }


		Tensor	_friendship;

	private:
		set<int> _winners;
		Tensor	_delta_w;
		Tensor	_delta_lw;

		LSOM1*	_lsom;
	};
}