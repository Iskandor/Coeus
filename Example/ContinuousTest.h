#pragma once
#include "SimpleContinuousEnv.h"
#include "CartPole.h"
#include <NeuralNetwork.h>

class ContinuousTest
{
public:
	ContinuousTest();
	~ContinuousTest();

	void run_simple_ddpg(int p_episodes);
	void run_simple_cacla(int p_episodes);
	void run_simple_cacer(int p_episodes);
	void run_cacla(int p_episodes);
	void run_ddpg(int p_episodes);
	int test_cart_pole(Coeus::NeuralNetwork& p_actor, Coeus::NeuralNetwork& p_critic, int p_episodes);
	bool evaluate_cart_pole(float p_reward);

private:
	vector<float> _rewards;
	SimpleContinuousEnv _environment;
	CartPole _cart_pole;

	void test_critic(Coeus::NeuralNetwork& p_network);
	void copy_state(vector<float>& p_observation, Tensor& p_state);
};

