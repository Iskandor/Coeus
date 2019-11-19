#pragma once
#include "NeuralNetwork.h"
#include "GradientAlgorithm.h"
#include "Gradient.h"

namespace Coeus
{
	class __declspec(dllexport) PolicyGradient
	{
	public:
		explicit PolicyGradient(NeuralNetwork* p_network);
		~PolicyGradient();

		Gradient& get_gradient(Tensor* p_state, int p_action, float p_delta) const;
		
	private:
		NeuralNetwork*	_network;
		NetworkGradient* _network_gradient;
	};
}

