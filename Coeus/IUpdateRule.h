#pragma once
#include <map>
#include "Tensor.h"
#include "NetworkGradient.h"

using namespace FLAB;
using namespace std;

namespace Coeus {

	class __declspec(dllexport) IUpdateRule
	{
	public:
		IUpdateRule(NetworkGradient* p_network_gradient, double p_alpha);
		virtual ~IUpdateRule();

		virtual void calc_update() = 0;
		virtual void merge(IUpdateRule** p_rule, int p_size) = 0;
		virtual IUpdateRule* clone(NetworkGradient* p_network_gradient) = 0;

		map<string, Tensor>* get_update() { return &_update; }

	protected:
		NetworkGradient* _network_gradient;

		double _alpha;

		map<string, Tensor> _update;		
	};
}

