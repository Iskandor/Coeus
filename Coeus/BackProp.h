#pragma once
#include "NeuralNetwork.h"
#include "ICostFunction.h"
#include "NetworkGradient.h"

namespace Coeus {

	class __declspec(dllexport) BackProp
	{
	public:
		BackProp(NeuralNetwork* p_network);
		~BackProp();

		void init(ICostFunction* p_cost_function, double p_alpha);

		double train(Tensor* p_input, Tensor* p_target);

	private:
		void calc_update();

		NeuralNetwork*	_network;
		ICostFunction*	_cost_function;
		NetworkGradient*	_network_gradient;

		map<string, Tensor> _update;

		double _alpha;
	};
}

