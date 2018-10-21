#pragma once
#include "IUpdateRule.h"

namespace Coeus {
	class __declspec(dllexport) AdadeltaRule : public IUpdateRule
	{
	public:
		AdadeltaRule(NetworkGradient* p_network_gradient, double p_alpha, double p_decay, double p_epsilon);
		~AdadeltaRule();

		void calc_update() override;
	private:
		void update_cache(const string& p_id, Tensor &p_gradient);
		void update_cache_delta(const string& p_id, Tensor &p_gradient);
		void init_structures() override;

		double _decay;
		double _epsilon;

		map<string, Tensor> _cache;
		map<string, Tensor> _cache_delta;
	};
}

