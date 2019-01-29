#pragma once
#include "IBatchModule.h"
#include "NeuralNetwork.h"
#include "IUpdateRule.h"

namespace Coeus {
	class __declspec(dllexport) SingleBatchModule : public IBatchModule
	{
	public:
		SingleBatchModule(NeuralNetwork* p_network, NetworkGradient* p_network_gradient, ICostFunction* p_cost_function, int p_batch);
		~SingleBatchModule();

		void run_batch(int p_b, int p_batch, vector<Tensor*>* p_input, vector<Tensor*>* p_target) override;

	private:
		NeuralNetwork*		_network;
		ICostFunction*		_cost_function;
		NetworkGradient*	_network_gradient;
	};
}
