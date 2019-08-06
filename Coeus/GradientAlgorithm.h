#pragma once
#include "NeuralNetwork.h"
#include "ICostFunction.h"
#include "NetworkGradient.h"
#include "IUpdateRule.h"

namespace Coeus {

	class __declspec(dllexport) GradientAlgorithm
	{
	public:
		GradientAlgorithm(NeuralNetwork* p_network);
		virtual ~GradientAlgorithm();

		virtual float train(Tensor* p_input, Tensor* p_target);
		virtual float train(vector<Tensor*>* p_input, Tensor* p_target);
		void reset() const;

	protected:
		void init(ICostFunction* p_cost_function, IUpdateRule* p_update_rule);
		
		NeuralNetwork*		_network;
		ICostFunction*		_cost_function;
		IUpdateRule*		_update_rule;
		NetworkGradient*	_network_gradient;
	};
}

