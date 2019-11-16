#include "CartPoleAgent.h"
#include "NeuralNetwork.h"
#include "CoreLayer.h"
#include "QLearning.h"
#include "Encoder.h"
#include "CACLA.h"
#include "TD.h"
#include <thread>
#include "LinearInterpolation.h"

using namespace Coeus;

CartPoleAgent::CartPoleAgent()
= default;


CartPoleAgent::~CartPoleAgent()
= default;

void CartPoleAgent::run(const boost::shared_ptr<Gym::Client>& p_client, const std::string& p_env_id, const int p_episodes) const
{
	NeuralNetwork network_critic;
	network_critic.add_layer(new CoreLayer("hidden0", 16, TANH, new TensorInitializer(UNIFORM, -0.01, 0.01), STATE));
	network_critic.add_layer(new CoreLayer("output", 1, TANH, new TensorInitializer(UNIFORM, -0.01, 0.01)));
	network_critic.add_connection("hidden0", "output");
	network_critic.init();

	TD critic(&network_critic, ADAM_RULE, 1e-3f, 0.9);

	NeuralNetwork network_actor;
	network_actor.add_layer(new CoreLayer("hidden0", 16, TANH, new TensorInitializer(UNIFORM, -0.01, 0.01), STATE));
	network_actor.add_layer(new CoreLayer("output", ACTION, TANH, new TensorInitializer(UNIFORM, -0.01, 0.01)));
	network_actor.add_connection("hidden0", "output");
	network_actor.init();

	CACLA actor(&network_actor, ADAM_RULE, 1e-3f);

	boost::shared_ptr<Gym::Environment> env = p_client->make(p_env_id);
	boost::shared_ptr<Gym::Space> action_space = env->action_space();
	boost::shared_ptr<Gym::Space> observation_space = env->observation_space();

	vector<float> a(ACTION);
	Tensor action({ ACTION }, Tensor::ZERO);
	Tensor state0({ STATE }, Tensor::ZERO);
	Tensor state1({ STATE }, Tensor::ZERO);

	int lost = 0;
	int win = 0;

	LinearInterpolation interpolation(0.1, 0.01, p_episodes);

	for (int e = 0; e < p_episodes; ++e) {
		printf("%s episode %i...\n", p_env_id.c_str(), e);
		Gym::State s;
		env->reset(&s);

		copy_state(s.observation, state0);

		float total_reward = 0;
		int total_steps = 0;

		float sigma = interpolation.interpolate(e);

		while (1) {
			network_critic.activate(&state0);
			network_actor.activate(&state0);
			action = actor.get_action(&state0, sigma);
			a[0] = action[0];
			env->step(a, false, &s);
			copy_state(s.observation, state1);

			total_reward += s.reward;
			total_steps += 1;

			float delta = critic.train(&state0, &state1, s.reward, s.done);
			actor.train(&state0, &action, delta);

			state0 = state1;

			this_thread::sleep_for(std::chrono::milliseconds(10));

			if (s.done) break;
		}

		if (total_steps >= 100) win++;
		else lost++;

		printf("%s episode %i finished in %i steps with reward %0.2f\n", p_env_id.c_str(), e, total_steps, total_reward);
		printf("%i / %i\n", win, lost);
	}
}

void CartPoleAgent::copy_state(vector<float>& p_observation, Tensor &p_state)
{
	for(int i = 0; i < p_observation.size(); i++)
	{
		p_state.set(i, p_observation[i]);
	}
}
