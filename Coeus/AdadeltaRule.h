#pragma once
#include "IUpdateRule.h"

namespace Coeus {
	class __declspec(dllexport) AdadeltaRule : public IUpdateRule
	{
	public:
		AdadeltaRule(NetworkGradient* p_network_gradient, double p_alpha, double p_decay = 0.9, double p_epsilon = 1e-8);
		~AdadeltaRule();
		IUpdateRule* clone(NetworkGradient* p_network_gradient) override;

		void calc_update(map<string, Tensor>* p_gradient) override;
		void reset() override;

	private:
		void update_cache(const string& p_id, Tensor &p_gradient);
		void update_cache_delta(const string& p_id, Tensor &p_gradient);

		double _decay;
		double _epsilon;

		map<string, Tensor> _cache;
		map<string, Tensor> _cache_delta;
	};
}

