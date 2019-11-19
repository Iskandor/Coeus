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
		ICM(NeuralNetwork* p_forward_model, GradientAlgorithm* p_forward_algorithm, int p_size = 0);
		~ICM();

		void activate(Tensor* p_state0, Tensor* p_action);
		float train(Tensor* p_state0, Tensor* p_action, Tensor* p_state1);
		void add(Tensor* p_state0, Tensor* p_action, Tensor* p_state1) const;
		float train(int p_sample);
		float get_intrinsic_reward(Tensor* p_state0, Tensor* p_action, Tensor* p_state1, float p_eta = 1);

		Tensor* get_output() const { return _forward_model->get_output(); }

	private:
		ReplayBuffer<TransitionItem>* _buffer;
		int		_sample;
		Tensor* _input;
		Tensor* _target;

		NeuralNetwork* _forward_model;
		GradientAlgorithm* _forward_algorithm;

		float _forward_reward;

		QuadraticCost _L;
	};
}
