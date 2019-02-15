#pragma once
#include "NeuralNetwork.h"
#include "GradientAlgorithm.h"

namespace Coeus {

	class __declspec(dllexport) BPTT
	{
	public:
		BPTT (NeuralNetwork* p_network, GradientAlgorithm* p_gradient_algorithm);
		~BPTT();

		float train(vector<Tensor*>* p_input, Tensor* p_target) const;

	private:
		NeuralNetwork* _network;
		GradientAlgorithm* _gradient_algorithm;
	};
}

