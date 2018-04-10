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
		void update_friendship();

		LSOM* lsom() const { return _lsom; }

	private:
		set<int> _winners;
		Tensor	_friendship;
		Tensor	_delta_w;
		Tensor	_delta_lw;

		LSOM*	_lsom;
	};
}