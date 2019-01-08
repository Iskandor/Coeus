#pragma once
#include "IUpdateRule.h"
#include "ADAMRule.h"

namespace Coeus
{
	class __declspec(dllexport) QLearningRule : public IUpdateRule
	{
	public:
		QLearningRule(NetworkGradient* p_network_gradient, double p_alpha, double p_gamma, double p_lambda);
		~QLearningRule();

		void calc_update(map<string, Tensor>* p_gradient, double p_delta);
		IUpdateRule* clone(NetworkGradient* p_network_gradient) override;
		void reset() override;
		void reset_traces();

	private:		
		void update_traces(map<string, Tensor>* p_gradient);
		double _gamma;
		double _lambda;

		map<string, Tensor> _e_traces;

		ADAMRule* _rule;
	};
}


