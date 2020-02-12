#pragma once
#include "NeuralNetwork.h"
#include "ICritic.h"

namespace Coeus
{
	class __declspec(dllexport) QLearning : public ICritic
	{
	public:
		QLearning(NeuralNetwork* p_network, GRADIENT_RULE p_grad_rule, float p_alpha, float p_gamma);
		virtual ~QLearning();

		float train(Tensor* p_state0, Tensor* p_action, Tensor* p_state1, float p_reward, bool p_final) override;

	protected:
		float _gamma;

	private:
		float calc_max_qa(Tensor* p_state) const;
	
	};
}
