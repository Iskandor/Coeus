#pragma once
#include "NeuralNetwork.h"
#include "BaseGradientAlgorithm.h"

namespace Coeus {

	class __declspec(dllexport) BPTT
	{
	public:
		BPTT (NeuralNetwork* p_network, BaseGradientAlgorithm* p_gradient_algorithm);
		~BPTT();

		double train(vector<Tensor*>* p_input, Tensor* p_target) const;

	private:
		NeuralNetwork* _network;
		BaseGradientAlgorithm* _gradient_algorithm;
	};
}

