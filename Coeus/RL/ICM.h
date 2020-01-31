/*
 * Intrinsic curiosity module
 * Curiosity-driven Exploration by Self-supervised Prediction
 * https://pathak22.github.io/noreward-rl/resources/icml17.pdf
 */
#pragma once
#include "NeuralNetwork.h"
#include "GradientAlgorithm.h"
#include "QuadraticCost.h"
#include "ReplayBuffer.h"
#include "BufferItems.h"

namespace Coeus {

	class __declspec(dllexport) ICM
	{
	public:
		ICM(NeuralNetwork* p_forward_model, NeuralNetwork* p_inverse_model, NeuralNetwork* p_feature_extractor, GRADIENT_RULE p_rule, float p_alpha, int p_size = 0, float p_beta = .2f);
		~ICM();

		void activate(Tensor* p_state0, Tensor* p_action, Tensor* p_state1, bool p_inverse_model = false);
		float train(Tensor* p_state0, Tensor* p_action, Tensor* p_state1);
		void add(Tensor* p_state0, Tensor* p_action, Tensor* p_state1) const;
		float train(int p_sample);
		float get_intrinsic_reward(Tensor* p_state0, Tensor* p_action, Tensor* p_state1, float p_eta = 1);

		Tensor* get_output() const { return _forward_model->get_output(); }

	private:
		ReplayBuffer<TransitionItem>* _buffer;
		int		_sample;

		Tensor* _fm_input;
		Tensor* _im_input;		
		Tensor* _fm_target;
		Tensor* _im_target;
		Tensor* _fe_input_s0;
		Tensor* _fe_input_s1;

		NeuralNetwork* _forward_model;
		NeuralNetwork* _inverse_model;
		NeuralNetwork* _feature_extractor;
		
		NetworkGradient* _fm_gradient;
		NetworkGradient* _im_gradient;
		NetworkGradient* _fe_gradient;

		IUpdateRule*	_fm_rule;
		IUpdateRule*	_im_rule;
		IUpdateRule*	_fe_rule;

		float _forward_reward;

		QuadraticCost _L;

		float _beta;
	};
}
