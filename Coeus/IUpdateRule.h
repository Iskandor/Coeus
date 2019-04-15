#pragma once
#include <map>
#include "Tensor.h"
#include "NetworkGradient.h"

using namespace std;

namespace Coeus {

	class __declspec(dllexport) IUpdateRule
	{
	public:
		IUpdateRule(NetworkGradient* p_network_gradient, float p_alpha);
		virtual ~IUpdateRule();

		virtual void calc_update(map<string, Tensor>* p_gradient, float p_alpha = 0);
		virtual IUpdateRule* clone(NetworkGradient* p_network_gradient) = 0;
		virtual void reset() = 0;

		map<string, Tensor>* get_update() { return &_update; }

	protected:
		float _alpha;
		NetworkGradient* _network_gradient;		
		map<string, Tensor> _update;
		
	};
}

