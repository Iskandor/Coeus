#pragma once
#include "IBatchModule.h"
#include "NeuralNetwork.h"
#include "NetworkGradient.h"
#include "IUpdateRule.h"
#include "ICostFunction.h"

namespace Coeus
{
	class __declspec(dllexport) OpenMPBatchModule : public IBatchModule
	{
	public:
		OpenMPBatchModule(NeuralNetwork* p_network, ICostFunction* p_cost_function, int p_batch);
		~OpenMPBatchModule();

		void run_batch(int p_b, int p_batch, vector<Tensor*>* p_input, vector<Tensor*>* p_target) override;
		float get_error(vector<Tensor*>* p_input, vector<Tensor*>* p_target) override;

	private:

		float*				_error;
		ICostFunction*		_cost_function;
		NeuralNetwork*		_network;
		NeuralNetwork**		_clone_network;
		NetworkGradient**	_network_gradient;
	};

}


