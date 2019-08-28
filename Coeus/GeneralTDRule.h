#pragma once
#include "IUpdateRule.h"

namespace Coeus
{
	class __declspec(dllexport) GeneralTDRule : public IUpdateRule
	{
	public:
		GeneralTDRule(NetworkGradient* p_network_gradient, IUpdateRule* p_rule, float p_alpha, float p_gamma, float p_lambda);
		~GeneralTDRule();

		void calc_update(map<string, Tensor>* p_gradient, float p_delta, float p_alpha);
		IUpdateRule* clone(NetworkGradient* p_network_gradient) override;
		void reset_traces();

	private:
		void calc_update(map<string, Tensor>* p_gradient, float p_alpha) override;
		void update_traces(map<string, Tensor>* p_gradient);
		float _gamma;
		float _delta;
		float _lambda;

		map<string, Tensor> _e_traces;

		IUpdateRule*	_rule;
	};
}


