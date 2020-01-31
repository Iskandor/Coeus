#pragma once
#include "NeuralNetwork.h"
#include "NetworkGradient.h"
#include "IUpdateRule.h"
#include "IMotivationModule.h"
#include "QuadraticCost.h"
#include "BufferItems.h"
#include "ReplayBuffer.h"

namespace Coeus {

	class __declspec(dllexport) GM2
	{
	public:
		GM2(NeuralNetwork* p_autoencoder, GRADIENT_RULE p_rule, float p_alpha, int p_size = 0);
		~GM2();

		void add(Tensor* p_state0, Tensor* p_action, Tensor* p_state1) const;
		float train(Tensor* p_state);
		float train(int p_sample);

		float uncertainty_motivation(Tensor* p_state, float p_eta = 1);
		
	private:
		void activate(Tensor* p_state) const;

		ReplayBuffer<TransitionItem>* _buffer;
		
		QuadraticCost		_mse;
		NeuralNetwork*		_autoencoder;
		NetworkGradient*	_gradient;
		IUpdateRule*		_rule;
	};
}
