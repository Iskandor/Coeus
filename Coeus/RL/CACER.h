#pragma once
#include "NeuralNetwork.h"
#include "CACLA.h"
#include "ReplayBuffer.h"
#include "BufferItems.h"

// Continuous Actor-Critic with Experience Replay (CACER)
// http://proceedings.mlr.press/v101/wang19a/wang19a.pdf

namespace Coeus
{
	class __declspec(dllexport) CACER : public CACLA
	{
	public:
		CACER(NeuralNetwork* p_critic, GRADIENT_RULE p_critic_rule, float p_critic_alpha, float p_gamma,
			  NeuralNetwork* p_actor, GRADIENT_RULE p_actor_rule, float p_actor_alpha, 
			  int p_buffer_size, int p_sample_size);
		~CACER();

		void train(Tensor* p_state0, Tensor* p_action0, Tensor* p_state1, float p_reward, bool p_final) override;
		
	private:
		ReplayBuffer<DQItem> *_buffer;
		int	_sample_size;
	};
}