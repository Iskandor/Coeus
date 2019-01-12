#pragma once
#include "IUpdateRule.h"

namespace Coeus {
	class __declspec(dllexport) PowerSignRule : public IUpdateRule
	{
	public:
		PowerSignRule(NetworkGradient* p_network_gradient, double p_alpha);
		~PowerSignRule();

		void calc_update(map<string, Tensor>* p_gradient) override;
		IUpdateRule* clone(NetworkGradient* p_network_gradient) override;
		void reset() override;

	private:
		map<string, Tensor> _m;
	};
}
