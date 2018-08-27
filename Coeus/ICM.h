#pragma once
#include "NeuralNetwork.h"
#include "BaseGradientAlgorithm.h"
#include "QuadraticCost.h"

namespace Coeus {

	class __declspec(dllexport) ICM
	{
	public:
		ICM(NeuralNetwork* p_forward_model, BaseGradientAlgorithm* p_forward_alogrithm, NeuralNetwork* p_inverse_model, BaseGradientAlgorithm* p_inverse_alogrithm);
		~ICM();

		double train(Tensor* p_state0, int p_action0, Tensor* p_state1);
		double get_intrinsic_reward(double p_beta = .5, double p_eta = 1) const;

	private:
		NeuralNetwork* _forward_model;
		BaseGradientAlgorithm* _forward_alogrithm;
		NeuralNetwork* _inverse_model;
		BaseGradientAlgorithm* _inverse_alogrithm;

		double _forward_reward;
		double _inverse_reward;

		Tensor* _forward_model_input;
		Tensor* _inverse_model_input;

		QuadraticCost _L;
	};
}
