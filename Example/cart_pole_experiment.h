#pragma once
#include <map>

#include "cart_pole.h"
#include "DDPG.h"
#include "forward_model.h"
#include "metacritic.h"
#include "neural_network.h"

class cart_pole_experiment
{
public:
	cart_pole_experiment();
	~cart_pole_experiment();

	void run_ddpg(int p_experiment_id, int p_episodes);
	void run_ddpg_fm(int p_experiment_id, int p_episodes);
	void run_ddpg_su(int p_experiment_id, int p_episodes);

	void ddpg_visualize_agent(neural_network& p_actor, neural_network& p_critic);
	void ddpg_visualize_agent(neural_network& p_actor, neural_network& p_critic, forward_model& p_forward_model);
	void ddpg_visualize_agent(neural_network& p_actor, neural_network& p_critic, forward_model& p_forward_model, metacritic& p_metacritic);
	
	void save_visualization(std::string p_filename);
	
private:
	float ddpg_test_agent(DDPG& p_agent, tensor& p_log_values, int p_episode);
	
	cart_pole _env;

	int _resolution;
	std::vector<float> _x;
	std::vector<float> _x_dot;
	std::vector<float> _theta;
	std::vector<float> _theta_dot;

	tensor _states;

	std::map<std::string, std::vector<tensor>> _data_outputs;

	const std::string ACTION = "ACTION";
	const std::string VALUE = "VALUE";
	const std::string REWARD = "REWARD";
	const std::string PREDICTION_ERROR = "PREDICTION_ERROR";
	const std::string ERROR_ESTIMATION = "ERROR_ESTIMATION";
};
