#pragma once
#include "NeuralNetwork.h"
#include "ICostFunction.h"
#include "NetworkGradient.h"
#include "IUpdateRule.h"
#include "PPLBatchModule.h"
#include "WarmStartup.h"
#include "AnnealingScheduler.h"

namespace Coeus {

	class __declspec(dllexport) GradientAlgorithm
	{
	public:
		GradientAlgorithm(NeuralNetwork* p_network);
		virtual ~GradientAlgorithm();

		virtual float train(Tensor* p_input, Tensor* p_target);
		virtual float train(vector<Tensor*>* p_input, vector<Tensor*>* p_target, int p_batch);

		void add_learning_rate_module(ILearningRateModule* p_learning_rate_module);
	protected:
		void init(ICostFunction* p_cost_function, IUpdateRule* p_update_rule);
		
		NeuralNetwork*		_network;
		ICostFunction*		_cost_function;
		IUpdateRule*		_update_rule;
		NetworkGradient*	_network_gradient;		

	private:
		IBatchModule*			_batch_module;
		ILearningRateModule*	_learning_rate_module;
		
	};
}

