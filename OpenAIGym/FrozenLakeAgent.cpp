#include "FrozenLakeAgent.h"
#include "RandomGenerator.h"
#include "ADAM.h"
#include "CoreLayer.h"
#include "QuadraticCost.h"
#include "QLearning.h"
#include "BackProph.h"
#include "ExponentialInterpolation.h"
#include "EGreedyExploration.h"
#include "Encoder.h"
#include "CountModule.h"


FrozenLakeAgent::FrozenLakeAgent()
{
	_network.add_layer(new CoreLayer("hidden0", 64, TANH, new TensorInitializer(UNIFORM, -0.1, 0.1), STATE));
	_network.add_layer(new CoreLayer("hidden1", 32, TANH, new TensorInitializer(UNIFORM, -0.1, 0.1)));
	_network.add_layer(new CoreLayer("output", ACTION, TANH, new TensorInitializer(UNIFORM, -0.1, 0.1)));
	// feed-forward connections
	_network.add_connection("hidden0", "hidden1");
	_network.add_connection("hidden1", "output");
	_network.init();

	_agent = new QLearning(&_network, ADAM_RULE, 1e-4f, 0.99);
}


FrozenLakeAgent::~FrozenLakeAgent()
{
	delete _agent;
}

void FrozenLakeAgent::run(const boost::shared_ptr<Gym::Client>& p_client, const std::string& p_env_id, const int p_episodes) {
	boost::shared_ptr<Gym::Environment> env = p_client->make(p_env_id);
	boost::shared_ptr<Gym::Space> action_space = env->action_space();
	boost::shared_ptr<Gym::Space> observation_space = env->observation_space();

	vector<float> action(1);

	EGreedyExploration exploration(0.1, new ExponentialInterpolation(0.1, 0.02, p_episodes));

	Tensor state0({ STATE }, Tensor::ZERO);
	Tensor state1({ STATE }, Tensor::ZERO);
	CountModule count_module(STATE);

	int lost = 0;
	int win = 0;

	for (int e = 0; e < p_episodes; ++e) {
		printf("%s episode %i...\n", p_env_id.c_str(), e);
		Gym::State s;
		env->reset(&s);
		Encoder::one_hot(state0, s.observation[0]);
		float total_reward = 0;
		int total_steps = 0;
		while (1) {			
			_network.activate(&state0);

			action[0] = exploration.get_action(_network.get_output());
			env->step(action, false, &s);

			Encoder::one_hot(state1, s.observation[0]);
			//count_module.update(&state1);
			total_reward += s.reward;
			total_steps += 1;

			_agent->train(&state0, action[0], &state1, s.reward); // +count_module.uncertainty_motivation());

			state0 = state1;

			if (s.done) break;
		}

		if (s.reward == 1) win++;
		else lost++;

		exploration.update(e);
		printf("%s episode %i finished in %i steps with reward %0.2f\n", p_env_id.c_str(), e, total_steps, total_reward);
		printf("%i / %i\n", win, lost);
	}

}