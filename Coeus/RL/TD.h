#pragma once
#include "NeuralNetwork.h"
#include "GradientAlgorithm.h"
#include "BufferItems.h"
#include "ICritic.h"

namespace Coeus
{

	class __declspec(dllexport) TD : public ICritic
	{
	public:
		TD(NeuralNetwork* p_network, GRADIENT_RULE p_grad_rule, float p_alpha, float p_gamma);
		~TD();

		float train(Tensor* p_state0, Tensor* p_action, Tensor* p_state1, float p_reward, bool p_final) override;
		Tensor train(vector<DQItem*>* p_sample) const;

	private:
		float _gamma;
	};

}
