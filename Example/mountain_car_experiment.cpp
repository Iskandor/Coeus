#include "mountain_car_experiment.h"
#include "neural_network.h"
#include "tensor_initializer.h"
#include "adam.h"
#include "DDPG.h"
#include "continuous_exploration.h"
#include <iostream>
#include "CACLA.h"
#include "DQN.h"
#include "discrete_exploration.h"
#include "linear_interpolation.h"


mountain_car_experiment::mountain_car_experiment()
{
}


mountain_car_experiment::~mountain_car_experiment()
{
}

void mountain_car_experiment::simple_ddpg(const int p_epochs)
{
	neural_network actor;
	actor.add_layer(new dense_layer("hidden0", 64, activation_function::relu(), tensor_initializer::xavier_uniform(), { _simple_env.STATE_DIM() }));
	actor.add_layer(new dense_layer("hidden1", 32, activation_function::relu(), tensor_initializer::xavier_uniform()));
	actor.add_layer(new dense_layer("output", _simple_env.ACTION_DIM(), activation_function::tanh(), tensor_initializer::uniform(-3e-1f, 3e-1f)));
	actor.add_connection("hidden0", "hidden1");
	actor.add_connection("hidden1", "output");
	actor.init();

	adam actor_optimizer(&actor, 2e-4f);

	neural_network critic;
	critic.add_layer(new dense_layer("hidden0", 64, activation_function::relu(), tensor_initializer::xavier_uniform(), { _simple_env.STATE_DIM() }));
	critic.add_layer(new dense_layer("hidden1", 32, activation_function::relu(), tensor_initializer::xavier_uniform(), { _simple_env.ACTION_DIM() }));
	critic.add_layer(new dense_layer("output", 1, activation_function::tanh(), tensor_initializer::uniform(-3e-3f, 3e-3f)));
	critic.add_connection("hidden0", "hidden1");
	critic.add_connection("hidden1", "output");
	critic.init();

	adam critic_optimizer(&critic, 1e-3f);

	DDPG agent(&actor, &actor_optimizer, &critic, &critic_optimizer, 0.99f, 10000, 64);

	continuous_exploration exploration;
	//exploration.init_ounoise(_simple_env.ACTION_DIM());
	exploration.init_gaussian(1.0f);

	tensor state0({ 1 });

	map<string, tensor*> critic_input;

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

		if (e % 100 == 0)
		{
			for (int i = 0; i < 21; i++)
			{
				state0[0] = i * 0.5f;

				tensor& action = agent.get_action(&state0);
				critic_input["hidden0"] = &state0;
				critic_input["hidden1"] = &action;

				float reward = _simple_env.get_reward(state0);
				printf("State %0.4f value %0.4f reward %0.4f policy %0.4f\n", state0[0], critic.forward(critic_input)[0], reward, action[0]);
			}
			cout << endl;
			getchar();
		}
	}

	for (int i = 0; i < 21; i++)
	{
		state0[0] = i * 0.5f;

		tensor& action = agent.get_action(&state0);
		critic_input["hidden0"] = &state0;
		critic_input["hidden1"] = &action;

		float reward = _simple_env.get_reward(state0);
		printf("State %0.4f value %0.4f reward %0.4f policy %0.4f\n", state0[0], critic.forward(critic_input)[0], reward, action[0]);
	}
	cout << endl;
	getchar();
}

void mountain_car_experiment::simple_cacla(int p_epochs)
{
	neural_network actor;
	actor.add_layer(new dense_layer("hidden0", 256, activation_function::tanhexp(), tensor_initializer::xavier_uniform(), { _simple_env.STATE_DIM() }));
	actor.add_layer(new dense_layer("hidden1", 256, activation_function::tanhexp(), tensor_initializer::xavier_uniform()));
	actor.add_layer(new dense_layer("output", _simple_env.ACTION_DIM(), activation_function::tanh(), tensor_initializer::uniform(-3e-1f, 3e-1f)));
	actor.add_connection("hidden0", "hidden1");
	actor.add_connection("hidden1", "output");
	actor.init();

	adam actor_optimizer(&actor, 1e-4f);

	neural_network critic;
	critic.add_layer(new dense_layer("hidden0", 256, activation_function::tanhexp(), tensor_initializer::xavier_uniform(), { _simple_env.STATE_DIM() }));
	critic.add_layer(new dense_layer("hidden1", 256, activation_function::tanhexp(), tensor_initializer::xavier_uniform()));
	critic.add_layer(new dense_layer("output", 1, activation_function::tanh(), tensor_initializer::uniform(-3e-3f, 3e-3f)));
	critic.add_connection("hidden0", "hidden1");
	critic.add_connection("hidden1", "output");
	critic.init();

	adam critic_optimizer(&critic, 2e-4f);

	CACLA agent(&actor, &actor_optimizer, &critic, &critic_optimizer, 0.99f);

	continuous_exploration exploration;
	//exploration.init_ounoise(_simple_env.ACTION_DIM());
	exploration.init_gaussian(1.0f);

	tensor state0({ 1 });

	map<string, tensor*> critic_input;

	for (int e = 0; e < p_epochs; e++)
	{
		float test_reward = 0;
		float train_reward = 0;
		_simple_env.reset();

		while (!_simple_env.is_finished())
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

		if (e % 500 == 0)
		{
			for (int i = 0; i < 21; i++)
			{
				state0[0] = i * 0.5f;

				tensor& action = agent.get_action(&state0);
				tensor& value = critic.forward(&state0);

				float reward = _simple_env.get_reward(state0);
				printf("State %0.4f value %0.4f reward %0.4f policy %0.4f\n", state0[0], value[0], reward, action[0]);
			}
			cout << endl;
			getchar();
		}
	}

	for (int i = 0; i < 21; i++)
	{
		state0[0] = i * 0.5f;

		tensor& action = agent.get_action(&state0);
		tensor& value = critic.forward(&state0);

		float reward = _simple_env.get_reward(state0);
		printf("State %0.4f value %0.4f reward %0.4f policy %0.4f\n", state0[0], value[0], reward, action[0]);
	}
	cout << endl;
	getchar();
}

void mountain_car_experiment::simple_dqn(int p_epochs)
{
	neural_network critic;
	critic.add_layer(new dense_layer("hidden0", 64, activation_function::tanhexp(), tensor_initializer::xavier_uniform(), { _simple_env.STATE_DIM() }));
	critic.add_layer(new dense_layer("hidden1", 32, activation_function::tanhexp(), tensor_initializer::xavier_uniform()));
	critic.add_layer(new dense_layer("output", 2, activation_function::sigmoid(), tensor_initializer::uniform(-3e-3f, 3e-3f)));
	critic.add_connection("hidden0", "hidden1");
	critic.add_connection("hidden1", "output");
	critic.init();

	adam critic_optimizer(&critic, 3e-4f);

	DQN agent(&critic, &critic_optimizer, 0.99f, 10000, 64, 1000);

	discrete_exploration exploration(discrete_exploration::EGREEDY, 0.7f, new linear_interpolation(0.7f, 0.0f, p_epochs));

	tensor state0({ 1 });
	tensor action0({ 1 });

	map<string, tensor*> critic_input;

	for (int e = 0; e < p_epochs; e++)
	{
		float test_reward = 0;
		float train_reward = 0;
		_simple_env.reset();

		while (!_simple_env.is_finished())
		{
			tensor state = _simple_env.get_state();
			tensor action = agent.get_action(&state);
			action = exploration.explore(action);
			if (action[0] == 1)
			{				
				action0[0] = -1.f;
				_simple_env.do_action(action0);
			}
			else
			{
				action0[0] = 1.f;
				_simple_env.do_action(action0);
			}			

			tensor next_state = _simple_env.get_state();
			const float reward = _simple_env.get_reward();
			const bool final = _simple_env.is_finished();

			agent.train(&state, &action, &next_state, reward, final);
			train_reward += reward;
		}
		exploration.update(e);

		_simple_env.reset();
		while (!_simple_env.is_finished())
		{
			tensor state = _simple_env.get_state();
			tensor action = agent.get_action(&state);
			if (action[0] == 1)
			{
				action0[0] = -1.f;
				_simple_env.do_action(action0);
			}
			else
			{
				action0[0] = 1.f;
				_simple_env.do_action(action0);
			}
			_simple_env.do_action(action0);
			test_reward += _simple_env.get_reward();
		}
		cout << "Episode " << e << " train reward " << train_reward << " test reward " << test_reward << endl;
		//exploration.reset();

		if (e % 500 == 0 && false)
		{
			for (int i = 0; i < 21; i++)
			{
				state0[0] = i * 0.5f;

				tensor& action = agent.get_action(&state0);
				if (action[0] == 1)
				{
					action0[0] = -1.f;
					_simple_env.do_action(action0);
				}
				else
				{
					action0[0] = 1.f;
					_simple_env.do_action(action0);
				}
				tensor& value = critic.forward(&state0);

				float reward = _simple_env.get_reward(state0);
				printf("State %0.4f value %0.4f %0.4f reward %0.4f policy %0.4f\n", state0[0], value[0], value[1], reward, action0[0]);
			}
			cout << endl;
			getchar();
		}
	}

	for (int i = 0; i < 21; i++)
	{
		state0[0] = i * 0.5f;

		tensor& action = agent.get_action(&state0);
		if (action[0] > action[1])
		{
			action0[0] = -1.f;
			_simple_env.do_action(action0);
		}
		else
		{
			action0[0] = 1.f;
			_simple_env.do_action(action0);
		}
		tensor& value = critic.forward(&state0);

		float reward = _simple_env.get_reward(state0);
		printf("State %0.4f value %0.4f %0.4f reward %0.4f policy %0.4f\n", state0[0], value[0], value[1], reward, action0[0]);
	}
	cout << endl;
	getchar();
}

void mountain_car_experiment::run_ddpg(const int p_epochs)
{
	neural_network actor;
	actor.add_layer(new dense_layer("hidden0", 400, activation_function::relu(), tensor_initializer::xavier_uniform(), { _env.STATE_DIM() }));
	actor.add_layer(new dense_layer("hidden1", 300, activation_function::relu(), tensor_initializer::xavier_uniform()));
	actor.add_layer(new dense_layer("output", _env.ACTION_DIM(), activation_function::tanh(), tensor_initializer::uniform(-3e-1f,3e-1f)));
	actor.add_connection("hidden0", "hidden1");
	actor.add_connection("hidden1", "output");
	actor.init();

	adam actor_optimizer(&actor, 1e-4f);

	neural_network critic;
	critic.add_layer(new dense_layer("hidden0", 400, activation_function::relu(), tensor_initializer::xavier_uniform(), { _env.STATE_DIM() }));
	critic.add_layer(new dense_layer("hidden1", 300, activation_function::relu(), tensor_initializer::xavier_uniform(), { _env.ACTION_DIM() }));
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

void mountain_car_experiment::run_ddpg_fm(const int p_epochs)
{
	neural_network fm_network;
	fm_network.add_layer(new dense_layer("hidden0", 50, activation_function::relu(), tensor_initializer::xavier_uniform(), { _env.STATE_DIM() + _env.ACTION_DIM() }));
	fm_network.add_layer(new dense_layer("hidden1", 30, activation_function::relu(), tensor_initializer::xavier_uniform()));
	fm_network.add_layer(new dense_layer("output", _env.STATE_DIM(), activation_function::linear(), tensor_initializer::uniform(-3e-1f, 3e-1f)));
	fm_network.add_connection("hidden0", "hidden1");
	fm_network.add_connection("hidden1", "output");
	fm_network.init();

	adam fm_optimizer(&fm_network, 2e-4f);

	forward_model fm_motivation(&fm_network, &fm_optimizer);

	neural_network actor;
	actor.add_layer(new dense_layer("hidden0", 400, activation_function::relu(), tensor_initializer::xavier_uniform(), { _env.STATE_DIM() }));
	actor.add_layer(new dense_layer("hidden1", 300, activation_function::relu(), tensor_initializer::xavier_uniform()));
	actor.add_layer(new dense_layer("output", _env.ACTION_DIM(), activation_function::tanh(), tensor_initializer::uniform(-3e-1f, 3e-1f)));
	actor.add_connection("hidden0", "hidden1");
	actor.add_connection("hidden1", "output");
	actor.init();

	adam actor_optimizer(&actor, 1e-4f);

	neural_network critic;
	critic.add_layer(new dense_layer("hidden0", 400, activation_function::relu(), tensor_initializer::xavier_uniform(), { _env.STATE_DIM() }));
	critic.add_layer(new dense_layer("hidden1", 300, activation_function::relu(), tensor_initializer::xavier_uniform(), { _env.ACTION_DIM() }));
	critic.add_layer(new dense_layer("output", 1, activation_function::linear(), tensor_initializer::uniform(-3e-3f, 3e-3f)));
	critic.add_connection("hidden0", "hidden1");
	critic.add_connection("hidden1", "output");
	critic.init();

	adam critic_optimizer(&critic, 2e-4f);

	DDPG agent(&actor, &actor_optimizer, &critic, &critic_optimizer, 0.99f, 10000, 64);
	agent.add_motivation(&fm_motivation);

	continuous_exploration exploration;
	exploration.init_ounoise(_env.ACTION_DIM(), 0.4f, 0.2f);


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
			action.reshape({ _env.ACTION_DIM() });
			_env.do_action(action);

			tensor next_state = _env.get_state();
			const float reward = _env.get_reward();
			const bool final = _env.is_finished();

			fm_motivation.train(&state, &action, &next_state);
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
