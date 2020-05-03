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
#include "sgd.h"

mountain_car_experiment::mountain_car_experiment()
{
	_resolution = 5;
	vector<tensor> state_list;

	for (int i = 0; i < _resolution; i++)
	{
		float position = i * 1.8f / (_resolution - 1) - 1.2f;
		float velocity = i * 0.14f / (_resolution - 1) - 0.07f;
		_position.push_back(position);
		_velocity.push_back(velocity);		
	}

	for(int i = 0; i < _resolution; i++)
	{		
		for (int j = 0; j < _resolution; j++)
		{
			tensor state({ 1,2 });
			state[0] = _position[j];
			state[1] = _velocity[i];
			state_list.push_back(state);
		}
	}

	tensor::concat(state_list, _states, 1);

	_data_outputs[ACTION] = std::vector<tensor>();
	_data_outputs[VALUE] = std::vector<tensor>();
	_data_outputs[REWARD] = std::vector<tensor>();
	_data_outputs[PREDICTION_ERROR] = std::vector<tensor>();
	_data_outputs[ERROR_ESTIMATION] = std::vector<tensor>();
}


mountain_car_experiment::~mountain_car_experiment()
{
}

void mountain_car_experiment::simple_ddpg(const int p_epochs)
{
	neural_network actor;
	actor.add_layer(new dense_layer("hidden0", 16, activation_function::relu(), tensor_initializer::xavier_uniform(), { _simple_env.STATE_DIM() }));
	actor.add_layer(new dense_layer("hidden1", 8, activation_function::relu(), tensor_initializer::xavier_uniform()));
	actor.add_layer(new dense_layer("output", _simple_env.ACTION_DIM(), activation_function::tanh(), tensor_initializer::uniform(-1e-3f, 1e-3f)));
	actor.add_connection("hidden0", "hidden1");
	actor.add_connection("hidden1", "output");
	actor.init();

	adam actor_optimizer(&actor, 1e-4f);

	neural_network critic;
	critic.add_layer(new dense_layer("hidden0", 16, activation_function::relu(), tensor_initializer::xavier_uniform(), { _simple_env.STATE_DIM() }));
	critic.add_layer(new dense_layer("hidden1", 8, activation_function::relu(), tensor_initializer::xavier_uniform(), { _simple_env.ACTION_DIM() }));
	critic.add_layer(new dense_layer("output", 1, activation_function::tanh(), tensor_initializer::uniform(-1e-4f, 1e-4f)));
	critic.add_connection("hidden0", "hidden1");
	critic.add_connection("hidden1", "output");
	critic.init();

	adam critic_optimizer(&critic, 2e-4f);

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
	actor.add_layer(new dense_layer("hidden0", 16, activation_function::tanhexp(), tensor_initializer::xavier_uniform(), { _simple_env.STATE_DIM() }));
	actor.add_layer(new dense_layer("hidden1", 8, activation_function::tanhexp(), tensor_initializer::xavier_uniform()));
	actor.add_layer(new dense_layer("output", _simple_env.ACTION_DIM(), activation_function::tanh(), tensor_initializer::uniform(-1e-1f, 1e-1f)));
	actor.add_connection("hidden0", "hidden1");
	actor.add_connection("hidden1", "output");
	actor.init();

	adam actor_optimizer(&actor, 1e-4f);

	neural_network critic;
	critic.add_layer(new dense_layer("hidden0", 16, activation_function::tanhexp(), tensor_initializer::xavier_uniform(), { _simple_env.STATE_DIM() }));
	critic.add_layer(new dense_layer("hidden1", 8, activation_function::tanhexp(), tensor_initializer::xavier_uniform()));
	critic.add_layer(new dense_layer("output", 1, activation_function::tanh(), tensor_initializer::uniform(-1e-4f, 1e-4f)));
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
			//cout << action[0] << " " << next_state[0] << " " << state[0] + action[0] * 0.1f << " " << reward << endl;
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

void mountain_car_experiment::run_cacla(int p_epochs)
{
	neural_network actor;
	actor.add_layer(new dense_layer("hidden0", 40, activation_function::relu(), tensor_initializer::xavier_uniform(), { _env.STATE_DIM() }));
	actor.add_layer(new dense_layer("hidden1", 30, activation_function::relu(), tensor_initializer::xavier_uniform()));
	actor.add_layer(new dense_layer("output", _env.ACTION_DIM(), activation_function::tanh(), tensor_initializer::uniform(-3e-1f, 3e-1f)));
	actor.add_connection("hidden0", "hidden1");
	actor.add_connection("hidden1", "output");
	actor.init();

	adam actor_optimizer(&actor, 1e-4f);

	neural_network critic;
	critic.add_layer(new dense_layer("hidden0", 40, activation_function::relu(), tensor_initializer::xavier_uniform(), { _env.STATE_DIM() }));
	critic.add_layer(new dense_layer("hidden1", 30, activation_function::relu(), tensor_initializer::xavier_uniform()));
	critic.add_layer(new dense_layer("output", 1, activation_function::linear(), tensor_initializer::uniform(-3e-3f, 3e-3f)));
	critic.add_connection("hidden0", "hidden1");
	critic.add_connection("hidden1", "output");
	critic.init();

	adam critic_optimizer(&critic, 2e-4f);

	CACLA agent(&actor, &actor_optimizer, &critic, &critic_optimizer, 0.99f);
	continuous_exploration exploration;
	exploration.init_ounoise(_env.ACTION_DIM(), 0.0f, 0.3f);
	//exploration.init_gaussian(1.0f);

	for (int e = 0; e < p_epochs; e++)
	{
		float test_reward = 0;
		float train_reward = 0;
		if (e % 10 == 0) {
			cacla_visualize_agent(actor, critic);
			//save_visualization("cacla_test");
			//getchar();
		}

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

	save_visualization("cacla_test");
}

void mountain_car_experiment::run_ddpg(int p_experiment_id, const int p_epochs)
{
	_data_outputs[ACTION].clear();
	_data_outputs[VALUE].clear();
	
	neural_network actor;
	actor.add_layer(new dense_layer("hidden0", 40, activation_function::relu(), tensor_initializer::xavier_uniform(), { _env.STATE_DIM() }));
	actor.add_layer(new dense_layer("hidden1", 30, activation_function::relu(), tensor_initializer::xavier_uniform()));
	actor.add_layer(new dense_layer("output", _env.ACTION_DIM(), activation_function::tanh(), tensor_initializer::uniform(-3e-1f,3e-1f)));
	actor.add_connection("hidden0", "hidden1");
	actor.add_connection("hidden1", "output");
	actor.init();

	adam actor_optimizer(&actor, 1e-4f);

	neural_network critic;
	critic.add_layer(new dense_layer("hidden0", 40, activation_function::relu(), tensor_initializer::xavier_uniform(), { _env.STATE_DIM() }));
	critic.add_layer(new dense_layer("hidden1", 30, activation_function::relu(), tensor_initializer::xavier_uniform(), { _env.ACTION_DIM() }));
	critic.add_layer(new dense_layer("output", 1, activation_function::linear(), tensor_initializer::uniform(-3e-3f, 3e-3f)));
	critic.add_connection("hidden0", "hidden1");
	critic.add_connection("hidden1", "output");
	critic.init();

	adam critic_optimizer(&critic, 2e-4f);

	DDPG agent(&actor, &actor_optimizer, &critic, &critic_optimizer, 0.99f, 10000, 64);

	continuous_exploration exploration;
	exploration.init_ounoise(_env.ACTION_DIM(), 0.0f, 0.3f);

	tensor log_rewards({ p_epochs });
	
	for (int e = 0; e < p_epochs; e++)
	{
		float test_reward = 0;
		float train_reward = 0;
		ddpg_visualize_agent(actor, critic);

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
		log_rewards[e] = test_reward;
	}

	save_visualization("ddpg_baseline_" + to_string(p_experiment_id));
	tensor::save_numpy("ddpg_baseline_" + to_string(p_experiment_id) + ".log", log_rewards);
}

void mountain_car_experiment::run_ddpg_fm(int p_experiment_id, const int p_epochs)
{
	_data_outputs[ACTION].clear();
	_data_outputs[VALUE].clear();
	_data_outputs[REWARD].clear();
	
	neural_network fm_network;
	fm_network.add_layer(new dense_layer("hidden0", 50, activation_function::relu(), tensor_initializer::xavier_uniform(), { _env.STATE_DIM() + _env.ACTION_DIM() }));
	fm_network.add_layer(new dense_layer("hidden1", 30, activation_function::relu(), tensor_initializer::xavier_uniform()));
	fm_network.add_layer(new dense_layer("output", _env.STATE_DIM(), activation_function::linear(), tensor_initializer::uniform(-3e-3f, 3e-3f)));
	fm_network.add_connection("hidden0", "hidden1");
	fm_network.add_connection("hidden1", "output");
	fm_network.init();

	adam fm_optimizer(&fm_network, 2e-4f);

	forward_model fm_motivation(&fm_network, &fm_optimizer);

	neural_network actor;
	actor.add_layer(new dense_layer("hidden0", 40, activation_function::relu(), tensor_initializer::xavier_uniform(), { _env.STATE_DIM() }));
	actor.add_layer(new dense_layer("hidden1", 30, activation_function::relu(), tensor_initializer::xavier_uniform()));
	actor.add_layer(new dense_layer("output", _env.ACTION_DIM(), activation_function::tanh(), tensor_initializer::uniform(-3e-1f, 3e-1f)));
	actor.add_connection("hidden0", "hidden1");
	actor.add_connection("hidden1", "output");
	actor.init();

	adam actor_optimizer(&actor, 1e-4f);

	neural_network critic;
	critic.add_layer(new dense_layer("hidden0", 40, activation_function::relu(), tensor_initializer::xavier_uniform(), { _env.STATE_DIM() }));
	critic.add_layer(new dense_layer("hidden1", 30, activation_function::relu(), tensor_initializer::xavier_uniform(), { _env.ACTION_DIM() }));
	critic.add_layer(new dense_layer("output", 1, activation_function::linear(), tensor_initializer::uniform(-3e-3f, 3e-3f)));
	critic.add_connection("hidden0", "hidden1");
	critic.add_connection("hidden1", "output");
	critic.init();

	adam critic_optimizer(&critic, 2e-4f);

	DDPG agent(&actor, &actor_optimizer, &critic, &critic_optimizer, 0.99f, 10000, 64);
	agent.add_motivation(&fm_motivation);

	continuous_exploration exploration;
	exploration.init_ounoise(_env.ACTION_DIM(), 0.0f, 0.3f);

	tensor log_rewards({ p_epochs });
	
	for (int e = 0; e < p_epochs; e++)
	{
		float test_reward = 0;
		float train_reward = 0;
		ddpg_visualize_agent(actor, critic, fm_motivation);

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
			
			agent.train(&state, &action, &next_state, reward, final);
			fm_motivation.train(&state, &action, &next_state);
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
		log_rewards[e] = test_reward;
		cout << "Episode " << e << " train reward " << train_reward << " test reward " << test_reward << endl;
	}

	save_visualization("ddpg_fm_" + to_string(p_experiment_id));
	tensor::save_numpy("ddpg_fm_" + to_string(p_experiment_id) + ".log", log_rewards);
}

void mountain_car_experiment::run_ddpg_su(int p_experiment_id, int p_epochs)
{
	_data_outputs[ACTION].clear();
	_data_outputs[VALUE].clear();
	_data_outputs[REWARD].clear();
	_data_outputs[PREDICTION_ERROR].clear();
	_data_outputs[ERROR_ESTIMATION].clear();

	neural_network fm_network;
	fm_network.add_layer(new dense_layer("hidden0", 50, activation_function::relu(), tensor_initializer::xavier_uniform(), { _env.STATE_DIM() + _env.ACTION_DIM() }));
	fm_network.add_layer(new dense_layer("hidden1", 30, activation_function::relu(), tensor_initializer::xavier_uniform()));
	fm_network.add_layer(new dense_layer("output", _env.STATE_DIM(), activation_function::linear(), tensor_initializer::uniform(-3e-3f, 3e-3f)));
	fm_network.add_connection("hidden0", "hidden1");
	fm_network.add_connection("hidden1", "output");
	fm_network.init();

	adam fm_optimizer(&fm_network, 2e-4f);

	forward_model fm_motivation(&fm_network, &fm_optimizer);

	neural_network metacritic_network;
	metacritic_network.add_layer(new dense_layer("hidden0", 50, activation_function::relu(), tensor_initializer::xavier_uniform(), { _env.STATE_DIM() + _env.ACTION_DIM() }));
	metacritic_network.add_layer(new dense_layer("hidden1", 30, activation_function::relu(), tensor_initializer::xavier_uniform()));
	metacritic_network.add_layer(new dense_layer("output", 1, activation_function::linear(), tensor_initializer::uniform(-3e-3f, 3e-3f)));
	metacritic_network.add_connection("hidden0", "hidden1");
	metacritic_network.add_connection("hidden1", "output");
	metacritic_network.init();

	adam metacritic_optimizer(&metacritic_network, 2e-4f);

	metacritic metacritic_motivation(&metacritic_network, &metacritic_optimizer, &fm_motivation, 1e-2f);

	neural_network actor;
	actor.add_layer(new dense_layer("hidden0", 40, activation_function::relu(), tensor_initializer::xavier_uniform(), { _env.STATE_DIM() }));
	actor.add_layer(new dense_layer("hidden1", 30, activation_function::relu(), tensor_initializer::xavier_uniform()));
	actor.add_layer(new dense_layer("output", _env.ACTION_DIM(), activation_function::tanh(), tensor_initializer::uniform(-3e-1f, 3e-1f)));
	actor.add_connection("hidden0", "hidden1");
	actor.add_connection("hidden1", "output");
	actor.init();

	adam actor_optimizer(&actor, 1e-4f);

	neural_network critic;
	critic.add_layer(new dense_layer("hidden0", 40, activation_function::relu(), tensor_initializer::xavier_uniform(), { _env.STATE_DIM() }));
	critic.add_layer(new dense_layer("hidden1", 30, activation_function::relu(), tensor_initializer::xavier_uniform(), { _env.ACTION_DIM() }));
	critic.add_layer(new dense_layer("output", 1, activation_function::linear(), tensor_initializer::uniform(-3e-3f, 3e-3f)));
	critic.add_connection("hidden0", "hidden1");
	critic.add_connection("hidden1", "output");
	critic.init();

	adam critic_optimizer(&critic, 2e-4f);

	DDPG agent(&actor, &actor_optimizer, &critic, &critic_optimizer, 0.99f, 10000, 64);
	agent.add_motivation(&metacritic_motivation);

	continuous_exploration exploration;
	exploration.init_ounoise(_env.ACTION_DIM(), 0.0f, 0.3f);

	tensor log_rewards({ p_epochs });

	for (int e = 0; e < p_epochs; e++)
	{
		float test_reward = 0;
		float train_reward = 0;
		ddpg_visualize_agent(actor, critic, fm_motivation, metacritic_motivation);

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
		log_rewards[e] = test_reward;
		cout << "Episode " << e << " train reward " << train_reward << " test reward " << test_reward << endl;
	}

	save_visualization("ddpg_su_" + to_string(p_experiment_id));
	tensor::save_numpy("ddpg_su_" + to_string(p_experiment_id) + ".log", log_rewards);
}

void mountain_car_experiment::ddpg_visualize_agent(neural_network& p_actor, neural_network& p_critic)
{
	tensor& actions = p_actor.forward(&_states);
	tensor& values = p_critic.forward({ &_states, &actions });
	actions.reshape({ 1, _resolution * _resolution });
	values.reshape({ 1, _resolution * _resolution });
	_data_outputs[ACTION].push_back(actions);
	_data_outputs[VALUE].push_back(values);
}

void mountain_car_experiment::ddpg_visualize_agent(neural_network& p_actor, neural_network& p_critic, forward_model& p_forward_model)
{
	tensor& actions = p_actor.forward(&_states);	
	tensor& values = p_critic.forward({ &_states, &actions });
	tensor state({ 2 });
	tensor action({ 1 });
	vector<tensor> next_states_list;
	
	for(int i = 0; i < _resolution * _resolution; i++)
	{
		state[0] = _states[i * 2];
		state[1] = _states[i * 2 + 1];
		action[0] = actions[i];
		_env.set_state(state);
		_env.do_action(action);
		tensor next_state = _env.get_state();
		next_state.reshape({ 1, 2 });
		next_states_list.push_back(next_state);
	}

	tensor next_states;
	tensor::concat(next_states_list, next_states, 1);
	
	tensor& rewards = p_forward_model.reward(&_states, &actions, &next_states);

	actions.reshape({ 1, _resolution * _resolution });
	values.reshape({ 1, _resolution * _resolution });
	rewards.reshape({ 1, _resolution * _resolution });	
	_data_outputs[ACTION].push_back(actions);
	_data_outputs[VALUE].push_back(values);
	_data_outputs[REWARD].push_back(actions);
}

void mountain_car_experiment::ddpg_visualize_agent(neural_network& p_actor, neural_network& p_critic, forward_model& p_forward_model, metacritic& p_metacritic)
{
	tensor& actions = p_actor.forward(&_states);
	tensor& values = p_critic.forward({ &_states, &actions });
	tensor state({ 2 });
	tensor action({ 1 });
	vector<tensor> next_states_list;

	for (int i = 0; i < _resolution * _resolution; i++)
	{
		state[0] = _states[i * 2];
		state[1] = _states[i * 2 + 1];
		action[0] = actions[i];
		_env.set_state(state);
		_env.do_action(action);
		tensor next_state = _env.get_state();
		next_state.reshape({ 1, 2 });
		next_states_list.push_back(next_state);
	}

	tensor next_states;
	tensor::concat(next_states_list, next_states, 1);

	tensor& error = p_forward_model.error(&_states, &actions, &next_states);
	tensor& error_estimate = p_metacritic.error(&_states, &actions);
	tensor& rewards = p_metacritic.reward(&_states, &actions, &next_states);
	
	actions.reshape({ 1, _resolution * _resolution });
	values.reshape({ 1, _resolution * _resolution });
	rewards.reshape({ 1, _resolution * _resolution });
	error.reshape({ 1, _resolution * _resolution });
	error_estimate.reshape({ 1, _resolution * _resolution });
	
	_data_outputs[ACTION].push_back(actions);
	_data_outputs[VALUE].push_back(values);
	_data_outputs[REWARD].push_back(rewards);
	_data_outputs[PREDICTION_ERROR].push_back(error);
	_data_outputs[ERROR_ESTIMATION].push_back(error_estimate);
}

void mountain_car_experiment::cacla_visualize_agent(neural_network& p_actor, neural_network& p_critic)
{
	tensor& actions = p_actor.forward(&_states);
	//tensor& values = p_critic.forward(&_states);
	
	tensor state({2});
	for(int i = 0; i < _resolution; i++)
	{
		for (int j = 0; j < _resolution; j++)
		{
			state[0] = _position[j];
			state[1] = _velocity[i];
			cout << p_critic.forward(&state)[0] << " ";
		}
		cout << endl;
	}
	cout << endl;


	tensor& values = p_critic.forward(&_states);
	values.reshape({ _resolution, _resolution });
	cout << values << endl;

	getchar();

	
	actions.reshape({ 1, _resolution * _resolution });
	values.reshape({ 1, _resolution * _resolution });
	_data_outputs[ACTION].push_back(actions);
	_data_outputs[VALUE].push_back(values);
}

void mountain_car_experiment::save_visualization(const std::string p_filename)
{
	tensor actions;
	tensor values;

	tensor::concat(_data_outputs[ACTION], actions, 1);
	tensor::concat(_data_outputs[VALUE], values, 1);

	tensor::save_numpy(p_filename + "_actions.npy", actions);
	tensor::save_numpy(p_filename + "_values.npy", values);

	if (!_data_outputs[REWARD].empty())
	{
		tensor rewards;
		tensor::concat(_data_outputs[REWARD], rewards, 1);
		tensor::save_numpy(p_filename + "_rewards.npy", rewards);
	}
	if (!_data_outputs[PREDICTION_ERROR].empty())
	{
		tensor prediction_error;
		tensor::concat(_data_outputs[PREDICTION_ERROR], prediction_error, 1);
		tensor::save_numpy(p_filename + "_prediction_errors.npy", prediction_error);
	}
	if (!_data_outputs[ERROR_ESTIMATION].empty())
	{
		tensor error_estimation;
		tensor::concat(_data_outputs[ERROR_ESTIMATION], error_estimation, 1);
		tensor::save_numpy(p_filename + "_error_estimations.npy", error_estimation);
	}
}
