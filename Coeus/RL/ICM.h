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
		ICM(NeuralNetwork* p_forward_model, NeuralNetwork* p_inverse_model, NeuralNetwork* p_head, GRADIENT_RULE p_rule, float p_alpha, int p_size = 0);
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
		
		Tensor* _target;

		NeuralNetwork* _forward_model;
		NeuralNetwork* _inverse_model;
		NeuralNetwork* _head;
		
		NetworkGradient* _fm_gradient;
		NetworkGradient* _im_gradient;
		NetworkGradient* _h_gradient;

		IUpdateRule*	_fm_rule;
		IUpdateRule*	_im_rule;
		IUpdateRule*	_h_rule;

		float _forward_reward;

		QuadraticCost _L;
	};
}
