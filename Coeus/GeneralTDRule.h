#pragma once
#include "IUpdateRule.h"

namespace Coeus
{
	class __declspec(dllexport) GeneralTDRule : public IUpdateRule
	{
	public:
		GeneralTDRule(NetworkGradient* p_network_gradient, IUpdateRule* p_rule, double p_alpha, double p_gamma, double p_lambda);
		~GeneralTDRule();

		void calc_update(map<string, Tensor>* p_gradient, double p_delta);
		IUpdateRule* clone(NetworkGradient* p_network_gradient) override;
		void reset_traces();

	private:		
		void update_traces(map<string, Tensor>* p_gradient);
		double _gamma;
		double _lambda;

		map<string, Tensor> _e_traces;

		IUpdateRule*	_rule;
	};
}


