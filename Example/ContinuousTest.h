#pragma once
#include "SimpleContinuousEnv.h"
#include "NeuralNetwork.h"
#include "CartPole.h"

class ContinuousTest
{
public:
	ContinuousTest();
	~ContinuousTest();

	void run(int p_hidden);
	void run_cart_pole(int p_episodes);
	void test_cart_pole(Coeus::NeuralNetwork& p_network);

private:
	SimpleContinuousEnv _environment;
	CartPole _cart_pole;

	void test_critic(Coeus::NeuralNetwork& p_network);
	void copy_state(vector<float>& p_observation, Tensor& p_state);
};

