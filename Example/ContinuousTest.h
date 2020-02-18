#pragma once
#include "SimpleContinuousEnv.h"
#include "CartPole.h"
#include <NeuralNetwork.h>
#include "MountainCar.h"
#include "Logger.h"

class ContinuousTest
{
public:
	ContinuousTest();
	~ContinuousTest();

	void run_simple_ddpg(int p_episodes);
	void run_simple_cacla(int p_episodes);
	void run_simple_cacer(int p_episodes);
	void run_cacla_cart_pole(int p_episodes, bool p_log = false);
	void run_ddpg_cart_pole(int p_episodes, bool p_log = false);
	
	void run_ddpg_mountain_car(const string& p_dir, int p_episodes, int p_hidden, float clr, float alr, bool p_log = false);
	void run_ddpg_mountain_car_icm(const string& p_dir, int p_episodes, int p_hidden, float clr, float alr, bool p_log = false);
	void run_ddpg_mountain_car_scm(const string& p_dir, int p_episodes, int p_hidden, float clr, float alr, bool p_log = false);
	
	float test_cart_pole(Coeus::NeuralNetwork& p_actor, Coeus::NeuralNetwork& p_critic, int p_episodes);
	float test_mountain_car(Coeus::NeuralNetwork& p_actor, Coeus::LoggerInstance* p_logger = nullptr);
	
	float evaluate_cart_pole(float p_reward);

private:
	vector<float> _rewards;
	SimpleContinuousEnv _environment;
	CartPole _cart_pole;
	MountainCar _mountain_car;

	void test_critic(Coeus::NeuralNetwork& p_network);
	void copy_state(vector<float>& p_observation, Tensor& p_state);
};

