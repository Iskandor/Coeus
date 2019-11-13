#pragma once
#include "IUpdateRule.h"

namespace Coeus {
	class __declspec(dllexport) AdadeltaRule : public IUpdateRule
	{
	public:
		AdadeltaRule(ParamModel* p_model, float p_alpha, float p_decay = 0.9, float p_epsilon = 1e-8);
		~AdadeltaRule();
		IUpdateRule* clone(ParamModel* p_model) override;

		void calc_update(map<string, Tensor>& p_gradient, float p_alpha = 0) override;

	private:
		void update_cache(const string& p_id, Tensor &p_gradient);
		void update_cache_delta(const string& p_id, Tensor &p_gradient);

		float _decay;
		float _epsilon;

		map<string, Tensor> _cache;
		map<string, Tensor> _cache_delta;
	};
}

