#pragma once
#include <map>
#include "Tensor.h"
#include "NetworkGradient.h"
#include "AnnealingScheduler.h"
#include "ILearningRateModule.h"

using namespace FLAB;
using namespace std;

namespace Coeus {

	class __declspec(dllexport) IUpdateRule
	{
	public:
		IUpdateRule(NetworkGradient* p_network_gradient, double p_alpha);
		virtual ~IUpdateRule();

		virtual void calc_update(map<string, Tensor>* p_gradient);
		virtual IUpdateRule* clone(NetworkGradient* p_network_gradient) = 0;
		virtual void reset() = 0;

		map<string, Tensor>* get_update() { return &_update; }

		void init_learning_rate_module(ILearningRateModule* p_learning_rate_module);

	protected:
		

		double _alpha;
		NetworkGradient* _network_gradient;
		ILearningRateModule* _learning_rate_module;
		map<string, Tensor> _update;
		
	};
}

