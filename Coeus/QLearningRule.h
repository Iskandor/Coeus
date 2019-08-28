#pragma once
#include "IUpdateRule.h"
#include "ADAMRule.h"

namespace Coeus
{
	class __declspec(dllexport) QLearningRule : public IUpdateRule
	{
	public:
		QLearningRule(NetworkGradient* p_network_gradient, float p_alpha, float p_gamma, float p_lambda);
		~QLearningRule();

		void calc_update(map<string, Tensor>* p_gradient, float p_delta);
		IUpdateRule* clone(NetworkGradient* p_network_gradient) override;
		void reset_traces();

	private:		
		void update_traces(map<string, Tensor>* p_gradient);
		float _gamma;
		float _lambda;

		map<string, Tensor> _e_traces;

		ADAMRule* _rule;
	};
}


