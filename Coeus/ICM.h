#pragma once
#include "NeuralNetwork.h"
#include "GradientAlgorithm.h"
#include "QuadraticCost.h"

namespace Coeus {

	class __declspec(dllexport) ICM
	{
	public:
		ICM(NeuralNetwork* p_forward_model, GradientAlgorithm* p_forward_alogrithm);
		~ICM();

		float train(Tensor* p_state0, Tensor* p_action, Tensor* p_state1);
		float get_intrinsic_reward(float p_eta = 1) const;

	private:
		NeuralNetwork* _forward_model;
		GradientAlgorithm* _forward_alogrithm;

		float _forward_reward;

		Tensor* _forward_model_input;

		QuadraticCost _L;
	};
}
