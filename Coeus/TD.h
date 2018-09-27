#pragma once
#include "NeuralNetwork.h"
#include "BaseGradientAlgorithm.h"

namespace Coeus
{

	class __declspec(dllexport) TD
	{
	public:
		TD(NeuralNetwork* p_network, BaseGradientAlgorithm* p_gradient_algorithm, double p_gamma, double p_alpha = 1);
		~TD();

		virtual double train(Tensor* p_state0, Tensor* p_state1, double p_reward);

	private:
		NeuralNetwork* _network;
		BaseGradientAlgorithm* _gradient_algorithm;
		double _alpha;
		double _gamma;

		Tensor _target;
	};

}
