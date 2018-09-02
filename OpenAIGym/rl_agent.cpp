#include "rl_agent.h"
#include "RandomGenerator.h"
#include "ADAM.h"
#include "InputLayer.h"
#include "CoreLayer.h"
#include "QuadraticCost.h"
#include "QLearning.h"
#include "BackProph.h"


rl_agent::rl_agent()
{
	_network.add_layer(new InputLayer("input", 16));
	_network.add_layer(new CoreLayer("hidden0", 16, RELU));
	_network.add_layer(new CoreLayer("output", 4, SIGMOID));

	_network.add_connection("input", "hidden0", Connection::UNIFORM, 0.01);
	_network.add_connection("hidden0", "output", Connection::UNIFORM, 0.01);
	_network.init();

	//BackProp optimizer(&_network);
	//optimizer.init(new QuadraticCost(), 0.01, 0.9, true);
	_optimizer = new BackProp(&_network);
	dynamic_cast<BackProp*>(_optimizer)->init(new QuadraticCost(), 0.1, 0.9);
	_agent = new QLearning(&_network, _optimizer, 0.99);
}


rl_agent::~rl_agent()
{
	delete _agent;
}

void rl_agent::run(const boost::shared_ptr<Gym::Client>& p_client, const std::string& p_env_id, const int p_episodes) {
	boost::shared_ptr<Gym::Environment> env = p_client->make(p_env_id);
	boost::shared_ptr<Gym::Space> action_space = env->action_space();
	boost::shared_ptr<Gym::Space> observation_space = env->observation_space();

	double epsilon = 0.5;
	vector<float> action(1);

	for (int e = 0; e < p_episodes; ++e) {
		printf("%s episode %i...\n", p_env_id.c_str(), e);
		Gym::State s;
		env->reset(&s);
		float total_reward = 0;
		int total_steps = 0;
		while (1) {
			Tensor state0 = encode_state(s.observation);
			_network.activate(&state0);

			action[0] = choose_action(_network.get_output(), epsilon);
			env->step(action, false, &s);

			Tensor state1 = encode_state(s.observation);
			total_reward += s.reward;
			total_steps += 1;

			_agent->train(&state0, action[0], &state1, s.reward);

			if (s.done) break;
		}
		printf("%s episode %i finished in %i steps with reward %0.2f\n", p_env_id.c_str(), e, total_steps, total_reward);

		if (epsilon > 0.1) {
			epsilon -= (1.0 / p_episodes);
		}
	}

}

Tensor rl_agent::encode_state(vector<float> &p_sensors) {
	const Tensor res({ STATE }, Tensor::ZERO);

	res[static_cast<int>(p_sensors[0])] = 1;

	return Tensor(res);
}

int rl_agent::choose_action(Tensor *p_input, const double p_epsilon) {
	int action;
	const double random = RandomGenerator::getInstance().random();

	if (random < p_epsilon) {
		action = RandomGenerator::getInstance().random(0, ACTION - 1);
	}
	else {
		action = p_input->max_value_index();
	}

	return action;
}

void rl_agent::binary_encoding(const double p_value, Tensor* p_vector) {
	p_vector->fill(0);
	(*p_vector)[p_value] = 1;
}
