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

		virtual void calc_update();
		map<string, Tensor>* get_update() { return &_update; }

	protected:
		virtual void init_structures();

		NetworkGradient* _network_gradient;
		bool _init_structures;

		double _alpha;

		map<string, Tensor> _update;
		map<string, Tensor> _update_batch;
	};
}

