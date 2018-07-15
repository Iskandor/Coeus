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
		void update();

		LSOM* lsom() const { return _lsom; }
		
	private:
		int _past;

		set<int> _winners;
		Tensor	_delta_b;
		Tensor	_delta_w;
		Tensor	_delta_lw;

		LSOM*	_lsom;

		int		_s;
		int		_t;
		vector<Tensor> _hist;

		Tensor	_avg;
		Tensor	_mean;
		Tensor	_deviation;
	};
}