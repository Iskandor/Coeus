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
		PPLBatchModule(NeuralNetwork* p_network, ICostFunction* p_cost_function, int p_batch);
		~PPLBatchModule();

		void run_batch(int p_b, int p_batch, vector<Tensor*>* p_input, vector<Tensor*>* p_target) override;
		float get_error(vector<Tensor*>* p_input, vector<Tensor*>* p_target) override;

	private:
		
		ICostFunction*			_cost_function;
		NeuralNetwork*			_network;
		NeuralNetwork**			_clone_network;
		NetworkGradient**		_network_gradient;
		vector<GradientAccumulator>	_gradient_accumulator_list;
	};
}
