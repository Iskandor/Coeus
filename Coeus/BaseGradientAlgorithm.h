#pragma once
#include "NeuralNetwork.h"
#include "ICostFunction.h"
#include "NetworkGradient.h"
#include "IUpdateRule.h"
#include "PPLBatchModule.h"

namespace Coeus {

	class __declspec(dllexport) BaseGradientAlgorithm
	{
	public:
		BaseGradientAlgorithm(NeuralNetwork* p_network);
		virtual ~BaseGradientAlgorithm();

		double train(Tensor* p_input, Tensor* p_target) const;
		double train(vector<Tensor*>* p_input, Tensor* p_target) const;
		double train(vector<Tensor*>* p_input, vector<Tensor*>* p_target, int p_batch);

	protected:
		double train(Tensor* p_target) const;
		void init(ICostFunction* p_cost_function, IUpdateRule* p_update_rule);
		
		NeuralNetwork*		_network;
		ICostFunction*		_cost_function;
		IUpdateRule*		_update_rule;
		NetworkGradient*	_network_gradient;

	private:
		IBatchModule*		_batch_module;
		
	};
}

