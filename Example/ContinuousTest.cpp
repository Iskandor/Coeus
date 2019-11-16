#include "ContinuousTest.h"
#include "NeuralNetwork.h"
#include "CoreLayer.h"
#include "TD.h"
#include "CACLA.h"
#include "Encoder.h"
#include "LinearInterpolation.h"
#include "LSTMLayer.h"
#include "DDPG.h"

using namespace Coeus;

ContinuousTest::ContinuousTest()
{
}


ContinuousTest::~ContinuousTest()
{
}

void ContinuousTest::run(int p_hidden)
{
	int input_dim = 1;
	NeuralNetwork network_critic;

	network_critic.add_layer(new CoreLayer("hidden0", p_hidden, RELU, new TensorInitializer(UNIFORM, -0.1, 0.1), input_dim));
	network_critic.add_layer(new CoreLayer("output", 1, SIGMOID, new TensorInitializer(UNIFORM, -0.1, 0.1)));
	// feed-forward connections	
	network_critic.add_connection("hidden0", "output");
	network_critic.init();

	TD critic(&network_critic, ADAM_RULE, 1e-2f, 0.9f);

	NeuralNetwork network_actor;

	network_actor.add_layer(new CoreLayer("hidden0", p_hidden, RELU, new TensorInitializer(UNIFORM, -0.1, 0.1), input_dim));
	network_actor.add_layer(new CoreLayer("output", 1, TANH, new TensorInitializer(UNIFORM, -0.1, 0.1)));
	// feed-forward connections
	network_actor.add_connection("hidden0", "output");
	network_actor.init();

	CACLA actor(&network_actor, ADAM_RULE, 1e-2f);

	float reward = 0;
	float delta = 0;
	Tensor action({ 1 }, Tensor::ZERO);
	Tensor state0({ input_dim }, Tensor::ZERO);
	Tensor state1({ input_dim }, Tensor::ZERO);
	int epoch = 1000;
	int win = 0;
	int lose = 0;

	

	for(int e = 0; e < epoch; e++)
	{
		int path = 0;
		_environment.reset();		
		//Encoder::pop_code(state0, _environment.get_state(), 0, 10);
		state0[0] = _environment.get_state();

		while (!_environment.is_finished())
		{
			//cout << _environment.get_state() << endl;
			action = actor.get_action(&state0, 0.1f);

			_environment.perform_action(action[0]);
			reward = _environment.get_reward();
			//Encoder::pop_code(state1, _environment.get_state(), 0, 10);
			state1[0] = _environment.get_state();

			delta = critic.train(&state0, &state1, reward, _environment.is_finished());
			actor.train(&state0, &action, delta);

			state0 = state1;
			path++;
		}

		if (_environment.is_winner()) win++;
		else lose++;

		cout << win << " / " << lose << "(" << path << ")" << endl;
	}

}

void ContinuousTest::run_cacla(const int p_episodes)
{
	int hidden = 128;
	float limit = 0.001f;

	NeuralNetwork network_critic;

	network_critic.add_layer(new CoreLayer("hidden0", hidden / 2, RELU, new TensorInitializer(GLOROT_UNIFORM), CartPole::STATE));
	network_critic.add_layer(new CoreLayer("hidden1", hidden / 4, TANH, new TensorInitializer(GLOROT_UNIFORM)));
	network_critic.add_layer(new CoreLayer("output", 1, TANH, new TensorInitializer(GLOROT_UNIFORM)));
	// feed-forward connections
	network_critic.add_connection("hidden0", "hidden1");
	network_critic.add_connection("hidden1", "output");
	network_critic.init();

	TD critic(&network_critic, ADAM_RULE, 1e-3f, 0.99);

	NeuralNetwork network_actor;

	network_actor.add_layer(new CoreLayer("hidden0", hidden, RELU, new TensorInitializer(GLOROT_UNIFORM), CartPole::STATE));
	network_actor.add_layer(new CoreLayer("hidden1", hidden / 2, TANH, new TensorInitializer(GLOROT_UNIFORM)));
	network_actor.add_layer(new CoreLayer("output", CartPole::ACTION, TANH, new TensorInitializer(GLOROT_UNIFORM)));
	// feed-forward connections
	network_actor.add_connection("hidden0", "hidden1");
	network_actor.add_connection("hidden1", "output");
	network_actor.init();

	CACLA actor(&network_actor, ADAM_RULE, 1e-4f);
	
	Tensor action({ CartPole::ACTION }, Tensor::ZERO);
	Tensor state0({ CartPole::STATE }, Tensor::ZERO);
	Tensor state1({ CartPole::STATE }, Tensor::ZERO);

	int lost = 0;
	int win = 0;

	LinearInterpolation interpolation(0.1, 0.1, p_episodes);

	for (int e = 0; e < p_episodes; ++e) {
		//printf("CartPole episode %i...\n", e);
		_cart_pole.reset();

		copy_state(_cart_pole.get_state(), state0);

		float total_reward = 0;
		int total_steps = 0;

		const float sigma = 1; //interpolation.interpolate(e);

		while (true) {
			action = actor.get_action(&state0, sigma);

			_cart_pole.perform_action(action[0]);
			//cout << network_actor.get_output()->at(0) << " " << action[0] << " " << _cart_pole.to_string();
			//cout << state0 << endl;
			copy_state(_cart_pole.get_state(), state1);

			total_reward += _cart_pole.get_reward();
			total_steps += 1;

			float delta = critic.train(&state0, &state1, _cart_pole.get_reward(), _cart_pole.is_finished());
			actor.train(&state0, &action, delta);

			state0 = state1;

			if (e % 1000 == 0) {
				//cout << network_critic.get_output()->at(0) << " " << delta << endl;
			}

			if (_cart_pole.is_finished()) {				
				break;
			}
		}

		if (total_steps >= 1000) win++;
		else lost++;

		if (e % 10 == 0) {			
			if (test_cart_pole(network_actor, network_critic, 6000) == 6000) {
				break;
			}
		}

		//printf("CartPole episode %i finished in %i steps with reward %0.2f\n", e, total_steps, total_reward);
		//printf("%i / %i\n", win, lost);
		
	}	
}

void ContinuousTest::run_ddpg(int p_episodes)
{
	int hidden = 128;
	float limit = 0.001f;

	NeuralNetwork network_critic;

	network_critic.add_layer(new CoreLayer("hidden0", hidden, TANH, new TensorInitializer(UNIFORM, -limit, limit), CartPole::STATE + CartPole::ACTION));
	network_critic.add_layer(new CoreLayer("output", 1, TANH, new TensorInitializer(UNIFORM, -limit, limit)));
	// feed-forward connections
	network_critic.add_connection("hidden0", "output");
	network_critic.init();
	
	NeuralNetwork network_actor;

	network_actor.add_layer(new CoreLayer("hidden0", hidden, TANH, new TensorInitializer(UNIFORM, -limit, limit), CartPole::STATE));
	network_actor.add_layer(new CoreLayer("output", CartPole::ACTION, TANH, new TensorInitializer(UNIFORM, -limit, limit)));
	// feed-forward connections
	network_actor.add_connection("hidden0", "output");
	network_actor.init();

	DDPG agent(&network_critic, ADAM_RULE, 1e-3f, 0.99, &network_actor, ADAM_RULE, 1e-4f, 10000, 64);

	Tensor action({ CartPole::ACTION }, Tensor::ZERO);
	Tensor state0({ CartPole::STATE }, Tensor::ZERO);
	Tensor state1({ CartPole::STATE }, Tensor::ZERO);

	int lost = 0;
	int win = 0;
	int total_steps = 0;

	LinearInterpolation interpolation(0.1, 0.1, p_episodes);

	for (int e = 0; e < p_episodes; ++e) {
		//printf("CartPole episode %i...\n", e);
		_cart_pole.reset();
		agent.reset();

		copy_state(_cart_pole.get_state(true), state0);

		float total_reward = 0;
		
		
		while (true) {
			action = agent.get_action(&state0, total_steps);

			_cart_pole.perform_action(action[0]);
			//network_critic.activate(&state0);
			//cout << network_critic.get_output()->at(0) << " " << network_actor.get_output()->at(0) << " " << action[0] << " " << _cart_pole.to_string() << endl;
			//cout << state0 << endl;
			copy_state(_cart_pole.get_state(true), state1);

			total_reward += _cart_pole.get_reward();
			total_steps += 1;

			agent.train(&state0, &action, &state1, _cart_pole.get_reward(), _cart_pole.is_finished());

			state0 = state1;

			if (e % 1000 == 0) {
				//cout << network_critic.get_output()->at(0) << " " << delta << endl;
			}

			if (_cart_pole.is_finished()) {
				break;
			}
		}

		//if (total_steps >= 1000) win++;
		//else lost++;

		if (e % 10 == 0) {
			/*
			cout << e << " : ";
			if (test_cart_pole(network_actor, network_critic, 6000) == 6000) {
				break;
			}
			*/
		}

		printf("CartPole episode %i finished in %i steps with reward %0.2f\n", e, total_steps, total_reward);
		//printf("%i / %i\n", win, lost);

	}
}

int ContinuousTest::test_cart_pole(Coeus::NeuralNetwork& p_actor, Coeus::NeuralNetwork& p_critic, int p_episodes)
{
	Tensor action({ CartPole::ACTION }, Tensor::ZERO);
	Tensor state0({ CartPole::STATE }, Tensor::ZERO);
	Tensor critic_input({ CartPole::STATE + CartPole::ACTION }, Tensor::ZERO);

	_cart_pole.reset();
	vector<float> obs = _cart_pole.get_state(true);
	copy_state(obs, state0);

	float total_reward = 0;
	int total_steps = 0;

	printf("CartPole test...\n");

	for (int e = 0; e < p_episodes; ++e) {
		//cout << state0 << endl;
		critic_input.reset_index();
		p_actor.activate(&state0);		

		critic_input.push_back(&state0);
		critic_input.push_back(p_actor.get_output());
		p_critic.activate(&critic_input);

		_cart_pole.perform_action(p_actor.get_output()->at(0));
		//cout << *p_actor.get_output() << " " << *p_critic.get_output() << " state: " << state0 << " reward: " << _cart_pole.get_reward() << endl;

		obs = _cart_pole.get_state(true);
		copy_state(obs, state0);

		total_reward += _cart_pole.get_reward();
		total_steps += 1;

		if (_cart_pole.is_finished()) {
			break;
		}
	}
	printf("CartPole test finished in %i steps with reward %0.2f\n", total_steps, total_reward);

	return total_steps;
}

void ContinuousTest::test_critic(NeuralNetwork& p_network)
{
	for(int i = 0; i < 11; i++)
	{
		
	}
}

void ContinuousTest::copy_state(vector<float>& p_observation, Tensor &p_state)
{
	for (int i = 0; i < p_observation.size(); i++)
	{
		p_state.set(i, p_observation[i]);
	}
}