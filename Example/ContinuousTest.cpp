#include "ContinuousTest.h"
#include "NeuralNetwork.h"
#include "CoreLayer.h"
#include "TD.h"
#include "CACLA.h"
#include "Encoder.h"
#include "LinearInterpolation.h"
#include "LSTMLayer.h"

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

void ContinuousTest::run_cart_pole(const int p_episodes)
{
	int hidden = 64;
	float limit = 0.1f;

	NeuralNetwork network_critic;

	network_critic.add_layer(new CoreLayer("hidden0", hidden, TANH, new TensorInitializer(UNIFORM, -limit, limit), CartPole::STATE));
	network_critic.add_layer(new CoreLayer("output", 1, TANH, new TensorInitializer(UNIFORM, -limit, limit)));
	// feed-forward connections
	network_critic.add_connection("hidden0", "output");
	network_critic.init();

	TD critic(&network_critic, BACKPROP_RULE, 1e-2f, 0.95);

	NeuralNetwork network_actor;

	network_actor.add_layer(new CoreLayer("hidden0", hidden, TANH, new TensorInitializer(UNIFORM, -limit, limit), CartPole::STATE));
	network_actor.add_layer(new CoreLayer("output", CartPole::ACTION, TANH, new TensorInitializer(UNIFORM, -limit, limit)));
	// feed-forward connections
	network_actor.add_connection("hidden0", "output");
	network_actor.init();

	CACLA actor(&network_actor, BACKPROP_RULE, 1e-3f);
	
	Tensor action({ CartPole::ACTION }, Tensor::ZERO);
	Tensor state0({ CartPole::STATE }, Tensor::ZERO);
	Tensor state1({ CartPole::STATE }, Tensor::ZERO);

	int lost = 0;
	int win = 0;

	LinearInterpolation interpolation(0.1, 0.1, p_episodes);

	for (int e = 0; e < p_episodes; ++e) {
		//printf("CartPole episode %i...\n", e);
		_cart_pole.reset();

		copy_state(_cart_pole.get_state(true), state0);

		float total_reward = 0;
		int total_steps = 0;

		const float sigma = 1; //interpolation.interpolate(e);

		while (true) {
			action = actor.get_action(&state0, sigma);

			_cart_pole.perform_action(action[0]);
			//cout << network_actor.get_output()->at(0) << " " << action[0] << " " << _cart_pole.to_string();
			//cout << state0 << endl;
			copy_state(_cart_pole.get_state(true), state1);

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

		if (total_steps >= 100) win++;
		else lost++;

		if (e % 1000 == 0) {
			test_cart_pole(network_actor, network_critic, 6000);
		}

		//printf("CartPole episode %i finished in %i steps with reward %0.2f\n", e, total_steps, total_reward);
		//printf("%i / %i\n", win, lost);
	}

	
}

void ContinuousTest::test_cart_pole(Coeus::NeuralNetwork& p_actor, Coeus::NeuralNetwork& p_critic, int p_episodes)
{
	Tensor action({ CartPole::ACTION }, Tensor::ZERO);
	Tensor state0({ CartPole::STATE }, Tensor::ZERO);
	Tensor state1({ CartPole::STATE }, Tensor::ZERO);

	_cart_pole.reset();
	copy_state(_cart_pole.get_state(true), state0);

	float total_reward = 0;
	int total_steps = 0;

	printf("CartPole test...\n");

	for (int e = 0; e < p_episodes; ++e) {


		p_actor.activate(&state0);
		p_critic.activate(&state0);

		action[0] = p_actor.get_output()->at(0);
		//cout << p_critic.get_output()->at(0) << " state: " << state0 << endl;

		_cart_pole.perform_action(action[0]);
		copy_state(_cart_pole.get_state(true), state1);

		total_reward += _cart_pole.get_reward();
		total_steps += 1;

		if (_cart_pole.is_finished()) {
			break;
		}
		else {
			state0 = state1;
		}
	}
	printf("CartPole test finished in %i steps with reward %0.2f\n", total_steps, total_reward);
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