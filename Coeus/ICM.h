#pragma once
#include "NeuralNetwork.h"
#include "BaseGradientAlgorithm.h"
#include "QuadraticCost.h"

namespace Coeus {

	class __declspec(dllexport) ICM
	{
	public:
		ICM(NeuralNetwork* p_forward_model, BaseGradientAlgorithm* p_forward_alogrithm);
		~ICM();

		double train(Tensor* p_state0, Tensor* p_action, Tensor* p_state1);
		double get_intrinsic_reward(double p_eta = 1) const;

	private:
		NeuralNetwork* _forward_model;
		BaseGradientAlgorithm* _forward_alogrithm;

		double _forward_reward;

		Tensor* _forward_model_input;

		QuadraticCost _L;
	};
}
