#pragma once
#include "NeuralNetwork.h"
#include "NetworkGradient.h"
#include "IUpdateRule.h"
#include <concrt.h>
#include "IBatchModule.h"

using namespace Concurrency;

namespace Coeus {
	class __declspec(dllexport) PPLBatchModule : public IBatchModule
	{
	public:
		PPLBatchModule(NeuralNetwork* p_network, IUpdateRule* p_update_rule, ICostFunction* p_cost_function, int p_batch);
		~PPLBatchModule();

		double run_batch(int p_b, int p_batch, vector<Tensor*>* p_input, vector<Tensor*>* p_target) override;

	private:
		
		double*				_error;
		ICostFunction*		_cost_function;
		NeuralNetwork*		_network;
		NeuralNetwork**		_clone_network;
		NetworkGradient**	_network_gradient;
		IUpdateRule*		_update_rule;
		
	};
}
