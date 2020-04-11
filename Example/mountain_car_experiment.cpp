#include "mountain_car_experiment.h"
#include "neural_network.h"
#include "tensor_initializer.h"
#include "adam.h"
#include "DDPG.h"
#include "continuous_exploration.h"
#include <iostream>


mountain_car_experiment::mountain_car_experiment()
{
}


mountain_car_experiment::~mountain_car_experiment()
{
}

void mountain_car_experiment::test_simple(const int p_epochs)
{
	neural_network actor;
	actor.add_layer(new dense_layer("hidden0", 64, activation_function::tanh(), tensor_initializer::lecun_uniform(), { _simple_env.STATE_DIM() }));
	actor.add_layer(new dense_layer("hidden1", 32, activation_function::tanh(), tensor_initializer::lecun_uniform()));
	actor.add_layer(new dense_layer("output", _simple_env.ACTION_DIM(), activation_function::tanh(), tensor_initializer::lecun_uniform()));
	actor.add_connection("hidden0", "hidden1");
	actor.add_connection("hidden1", "output");
	actor.init();

	adam actor_optimizer(&actor, 1e-3f);

	neural_network critic;
	critic.add_layer(new dense_layer("hidden0", 64, activation_function::tanh(), tensor_initializer::lecun_uniform(), { _simple_env.STATE_DIM() }));
	critic.add_layer(new dense_layer("hidden1", 32, activation_function::tanh(), tensor_initializer::lecun_uniform(), { _simple_env.ACTION_DIM() }));
	critic.add_layer(new dense_layer("output", 1, activation_function::linear(), tensor_initializer::lecun_uniform()));
	critic.add_connection("hidden0", "hidden1");
	critic.add_connection("hidden1", "output");
	critic.init();

	adam critic_optimizer(&critic, 2e-3f);

	DDPG agent(&actor, &actor_optimizer, &critic, &critic_optimizer, 0.99f, 10000, 64);

	continuous_exploration exploration;
	exploration.init_ounoise(_simple_env.ACTION_DIM());
	//exploration.init_gaussian(0.2f);

	tensor state0({ 1 });

	map<string, tensor*> critic_input;

	for (int i = 0; i < 21; i++)
	{
		state0[0] = i * 0.5f;

		tensor& action = agent.get_action(&state0);
		critic_input["hidden0"] = &state0;
		critic_input["hidden1"] = &action;
		critic.forward(critic_input);

		float reward = _simple_env.get_reward(state0);
		printf("State %0.4f value %0.4f reward %0.4f policy %0.4f\n", state0[0], critic.forward(critic_input)[0], reward, action[0]);
	}

	for(int e = 0; e < p_epochs; e++)
	{
		float test_reward = 0;
		float train_reward = 0;
		_simple_env.reset();

		while(!_simple_env.is_finished())
		{
			tensor state = _simple_env.get_state();
			tensor action = agent.get_action(&state);
			action = exploration.explore(action);
			_simple_env.do_action(action);

			tensor next_state = _simple_env.get_state();
			const float reward = _simple_env.get_reward();
			const bool final = _simple_env.is_finished();

			agent.train(&state, &action, &next_state, reward, final);
			train_reward += reward;
		}
		exploration.reset();

		_simple_env.reset();
		while (!_simple_env.is_finished())
		{			
			tensor state = _simple_env.get_state();
			tensor action = agent.get_action(&state);
			_simple_env.do_action(action);
			test_reward += _simple_env.get_reward();
		}
		cout << "Episode " << e << " train reward " << train_reward << " test reward " << test_reward << endl;
		//exploration.reset();
	}

	for (int i = 0; i < 21; i++)
	{
		state0[0] = i * 0.5f;

		tensor& action = agent.get_action(&state0);
		critic_input["hidden0"] = &state0;
		critic_input["hidden1"] = &action;
		critic.forward(critic_input);

		float reward = _simple_env.get_reward(state0);
		printf("State %0.4f value %0.4f reward %0.4f policy %0.4f\n", state0[0], critic.forward(critic_input)[0], reward, action[0]);
	}
	cout << endl;
}

void mountain_car_experiment::run_ddpg(const int p_epochs)
{
	neural_network actor;
	actor.add_layer(new dense_layer("hidden0", 400, activation_function::relu(), tensor_initializer::glorot_uniform(), { _env.STATE_DIM() }));
	actor.add_layer(new dense_layer("hidden1", 300, activation_function::relu(), tensor_initializer::glorot_uniform()));
	actor.add_layer(new dense_layer("output", _env.ACTION_DIM(), activation_function::tanh(), tensor_initializer::uniform(-3e-1f,3e-1f)));
	actor.add_connection("hidden0", "hidden1");
	actor.add_connection("hidden1", "output");
	actor.init();

	adam actor_optimizer(&actor, 1e-4f);

	neural_network critic;
	critic.add_layer(new dense_layer("hidden0", 400, activation_function::relu(), tensor_initializer::glorot_uniform(), { _env.STATE_DIM() }));
	critic.add_layer(new dense_layer("hidden1", 300, activation_function::relu(), tensor_initializer::glorot_uniform(), { _env.ACTION_DIM() }));
	critic.add_layer(new dense_layer("output", 1, activation_function::linear(), tensor_initializer::uniform(-3e-3f, 3e-3f)));
	critic.add_connection("hidden0", "hidden1");
	critic.add_connection("hidden1", "output");
	critic.init();

	adam critic_optimizer(&critic, 2e-4f);

	DDPG agent(&actor, &actor_optimizer, &critic, &critic_optimizer, 0.99f, 10000, 64);

	continuous_exploration exploration;
	exploration.init_ounoise(_env.ACTION_DIM(), 0.4f);


	for (int e = 0; e < p_epochs; e++)
	{
		float test_reward = 0;
		float train_reward = 0;
		_env.reset();

		while (!_env.is_finished())
		{
			tensor state = _env.get_state();
			tensor action = agent.get_action(&state);
			action = exploration.explore(action);
			_env.do_action(action);

			tensor next_state = _env.get_state();
			const float reward = _env.get_reward();
			const bool final = _env.is_finished();

			agent.train(&state, &action, &next_state, reward, final);
			train_reward += reward;
		}
		exploration.reset();

		_env.reset();
		while (!_env.is_finished())
		{
			tensor state = _env.get_state();
			tensor action = agent.get_action(&state);
			_env.do_action(action);
			test_reward += _env.get_reward();
		}
		cout << "Episode " << e << " train reward " << train_reward << " test reward " << test_reward << endl;
	}
}
