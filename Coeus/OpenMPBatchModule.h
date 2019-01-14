#pragma once
#include "IBatchModule.h"
#include "NeuralNetwork.h"
#include "IUpdateRule.h"

namespace Coeus
{
	class __declspec(dllexport) OpenMPBatchModule : public IBatchModule
	{
	public:
		OpenMPBatchModule(NeuralNetwork* p_network, ICostFunction* p_cost_function, int p_batch);
		~OpenMPBatchModule();

		double run_batch(int p_b, int p_batch, vector<Tensor*>* p_input, vector<Tensor*>* p_target) override;

	private:

		double*				_error;
		ICostFunction*		_cost_function;
		NeuralNetwork*		_network;
		NeuralNetwork**		_clone_network;
		NetworkGradient**	_network_gradient;
	};

}

