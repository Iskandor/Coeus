#pragma once
#include "forward_model.h"
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

	void run_ddpg(int p_epochs);
	void run_ddpg_fm(int p_epochs);

	void visualize_agent(neural_network& p_actor, neural_network& p_critic);
	void visualize_agent(neural_network& p_actor, neural_network& p_critic, forward_model& p_forward_model);
	void save_visualization(std::string p_filename);
	
private:
	int _resolution;
	std::vector<float> _position;
	std::vector<float> _velocity;
	tensor _states;

	std::vector<tensor> _actor_outputs;
	std::vector<tensor> _critic_outputs;
	std::vector<tensor> _fm_outputs;
	
	simple_continuous_env _simple_env;
	mountain_car _env;
};

