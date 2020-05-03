#pragma once
#include "forward_model.h"
#include "metacritic.h"
#include "simple_continuous_env.h"
#include "mountain_car.h"
#include "neural_network.h"

class mountain_car_experiment
{
public:
	mountain_car_experiment();
	~mountain_car_experiment();

	void simple_ddpg(int p_epochs);
	void simple_cacla(int p_epochs);
	void simple_dqn(int p_epochs);

	void run_cacla(int p_epochs);
	
	void run_ddpg(int p_experiment_id, int p_epochs);
	void run_ddpg_fm(int p_experiment_id, int p_epochs);
	void run_ddpg_su(int p_experiment_id, int p_epochs);

	void ddpg_visualize_agent(neural_network& p_actor, neural_network& p_critic);
	void ddpg_visualize_agent(neural_network& p_actor, neural_network& p_critic, forward_model& p_forward_model);
	void ddpg_visualize_agent(neural_network& p_actor, neural_network& p_critic, forward_model& p_forward_model, metacritic& p_metacritic);
	void cacla_visualize_agent(neural_network& p_actor, neural_network& p_critic);
	void save_visualization(std::string p_filename);
	
private:
	int _resolution;
	std::vector<float> _position;
	std::vector<float> _velocity;
	tensor _states;

	std::map<std::string, std::vector<tensor>> _data_outputs;
	
	simple_continuous_env _simple_env;
	mountain_car _env;

	const std::string ACTION = "ACTION";
	const std::string VALUE = "VALUE";
	const std::string REWARD = "REWARD";
	const std::string PREDICTION_ERROR = "PREDICTION_ERROR";
	const std::string ERROR_ESTIMATION = "ERROR_ESTIMATION";
};

