#include "maze_experiment.h"
#include "neural_network.h"
#include "adam.h"
#include "Qlearning.h"
#include "discrete_exploration.h"
#include "tensor_initializer.h"
#include "linear_interpolation.h"
#include <iostream>

#include "AC.h"
#include "SARSA.h"
#include "DQN.h"


maze_experiment::maze_experiment()
{
	int topology[] =
	{ 0, 0, 0, 0,
		0, 2, 0, 2,
		0, 0, 0, 2,
		2, 0, 0, 0 };

	_maze = new maze(topology, 4, 4, 15, false);

	/*
	int topology[] =
	{ 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 2, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 2, 0, 0,
	0, 0, 0, 2, 0, 0, 0, 0,
	0, 2, 2, 0, 0, 0, 2, 0,
	0, 2, 0, 0, 2, 0, 2, 0,
	0, 0, 0, 2, 0, 0, 0, 0
	};

	_maze = new Maze(topology, 8, 8, 63, false);
	*/
}


maze_experiment::~maze_experiment()
{
	delete _maze;
}

void maze_experiment::run_qlearning(const int p_episodes)
{
	neural_network critic;
	critic.add_layer(new dense_layer("hidden0", 64, activation_function::relu(), tensor_initializer::lecun_uniform(), { _maze->STATE_DIM() }));
	critic.add_layer(new dense_layer("hidden1", 16, activation_function::relu(), tensor_initializer::lecun_uniform()));
	critic.add_layer(new dense_layer("output", _maze->ACTION_DIM(), activation_function::sigmoid(), tensor_initializer::lecun_uniform()));
	critic.add_connection("hidden0", "hidden1");
	critic.add_connection("hidden1", "output");
	critic.init();

	adam critic_optimizer(&critic, 1e-3f);
	
	Qlearning agent(&critic, &critic_optimizer, 0.99f);

	discrete_exploration exploration(discrete_exploration::EGREEDY, 0.5f, new linear_interpolation(0.5f, 0.0f, p_episodes));

	int success = 0;
	int fail = 0;

	test_agent(critic);

	for (int e = 0; e < p_episodes; e++)
	{
		float test_reward = 0;
		float train_reward = 0;
		_maze->reset();
		tensor state = _maze->get_state();

		while (!_maze->is_finished())
		{			
			tensor action = agent.get_action(&state);
			action = exploration.explore(action);
			_maze->do_action(action);

			tensor next_state = _maze->get_state();
			const float reward = _maze->get_reward();
			const bool final = _maze->is_finished();

			agent.train(&state, &action, &next_state, reward, final);
			train_reward += reward;

			state = next_state;
		}		
		exploration.update(e);

		_maze->reset();
		while (!_maze->is_finished())
		{
			tensor state = _maze->get_state();
			tensor action = agent.get_action(&state);
			_maze->do_action(action);
			test_reward += _maze->get_reward();
		}
		if (_maze->get_reward() == 1) success++; else fail++;
		
		cout << "Episode " << e << " " << success << " / " << fail << endl;		
	}

	cout << _maze->to_string() << endl;
	test_agent(critic);
}

void maze_experiment::run_sarsa(int p_episodes)
{
	neural_network critic;
	critic.add_layer(new dense_layer("hidden0", 64, activation_function::tanhexp(), tensor_initializer::lecun_uniform(), { _maze->STATE_DIM() }));
	critic.add_layer(new dense_layer("hidden1", 16, activation_function::tanhexp(), tensor_initializer::lecun_uniform()));
	critic.add_layer(new dense_layer("output", _maze->ACTION_DIM(), activation_function::sigmoid(), tensor_initializer::lecun_uniform()));
	critic.add_connection("hidden0", "hidden1");
	critic.add_connection("hidden1", "output");
	critic.init();

	adam critic_optimizer(&critic, 1e-3f);

	SARSA agent(&critic, &critic_optimizer, 0.99f);

	discrete_exploration exploration(discrete_exploration::EGREEDY, 0.5f, new linear_interpolation(0.5f, 0.0f, p_episodes));

	int success = 0;
	int fail = 0;

	test_agent(critic);

	for (int e = 0; e < p_episodes; e++)
	{
		float test_reward = 0;
		float train_reward = 0;
		_maze->reset();
		tensor state = _maze->get_state();
		tensor action = agent.get_action(&state);
		action = exploration.explore(action);

		while (!_maze->is_finished())
		{
			_maze->do_action(action);

			tensor next_state = _maze->get_state();
			const float reward = _maze->get_reward();
			const bool final = _maze->is_finished();
			tensor next_action = agent.get_action(&next_state);
			next_action = exploration.explore(next_action);

			agent.train(&state, &action, &next_state, &next_action, reward, final);
			train_reward += reward;

			state = next_state;
			action = next_action;
		}
		exploration.update(e);

		_maze->reset();
		while (!_maze->is_finished())
		{
			tensor state = _maze->get_state();
			tensor action = agent.get_action(&state);
			_maze->do_action(action);
			test_reward += _maze->get_reward();
		}
		if (_maze->get_reward() == 1) success++; else fail++;

		cout << "Episode " << e << " " << success << " / " << fail << endl;
	}

	cout << _maze->to_string() << endl;
	test_agent(critic);
}

void maze_experiment::run_dqn(int p_episodes)
{
	neural_network critic;
	critic.add_layer(new dense_layer("hidden0", 64, activation_function::tanhexp(), tensor_initializer::lecun_uniform(), { _maze->STATE_DIM() }));
	critic.add_layer(new dense_layer("hidden1", 16, activation_function::tanhexp(), tensor_initializer::lecun_uniform()));
	critic.add_layer(new dense_layer("output", _maze->ACTION_DIM(), activation_function::sigmoid(), tensor_initializer::lecun_uniform()));
	critic.add_connection("hidden0", "hidden1");
	critic.add_connection("hidden1", "output");
	critic.init();

	adam critic_optimizer(&critic, 1e-3f);

	DQN agent(&critic, &critic_optimizer, 0.99f, 10000, 64, 1000);

	discrete_exploration exploration(discrete_exploration::EGREEDY, 0.5f, new linear_interpolation(0.5f, 0.0f, p_episodes));

	int success = 0;
	int fail = 0;

	test_agent(critic);

	for (int e = 0; e < p_episodes; e++)
	{
		float test_reward = 0;
		float train_reward = 0;
		_maze->reset();
		tensor state = _maze->get_state();

		while (!_maze->is_finished())
		{
			tensor action = agent.get_action(&state);
			action = exploration.explore(action);
			_maze->do_action(action);

			tensor next_state = _maze->get_state();
			const float reward = _maze->get_reward();
			const bool final = _maze->is_finished();

			agent.train(&state, &action, &next_state, reward, final);
			train_reward += reward;

			state = next_state;
		}
		exploration.update(e);

		_maze->reset();
		while (!_maze->is_finished())
		{
			tensor state = _maze->get_state();
			tensor action = agent.get_action(&state);
			_maze->do_action(action);
			test_reward += _maze->get_reward();
		}
		if (_maze->get_reward() == 1) success++; else fail++;

		cout << "Episode " << e << " " << success << " / " << fail << endl;
	}

	cout << _maze->to_string() << endl;
	test_agent(critic);
}

void maze_experiment::run_ac(int p_episodes)
{
	neural_network critic;
	critic.add_layer(new dense_layer("hidden0", 64, activation_function::tanhexp(), tensor_initializer::lecun_uniform(), { _maze->STATE_DIM() }));
	critic.add_layer(new dense_layer("hidden1", 16, activation_function::tanhexp(), tensor_initializer::lecun_uniform()));
	critic.add_layer(new dense_layer("output", 1, activation_function::sigmoid(), tensor_initializer::lecun_uniform()));
	critic.add_connection("hidden0", "hidden1");
	critic.add_connection("hidden1", "output");
	critic.init();

	adam critic_optimizer(&critic, 2e-4f);

	neural_network actor;
	actor.add_layer(new dense_layer("hidden0", 64, activation_function::tanhexp(), tensor_initializer::lecun_uniform(), { _maze->STATE_DIM() }));
	actor.add_layer(new dense_layer("hidden1", 16, activation_function::tanhexp(), tensor_initializer::lecun_uniform()));
	actor.add_layer(new dense_layer("output", _maze->ACTION_DIM(), activation_function::softmax(), tensor_initializer::lecun_uniform()));
	actor.add_connection("hidden0", "hidden1");
	actor.add_connection("hidden1", "output");
	actor.init();

	adam actor_optimizer(&actor, 1e-4f);

	AC agent(&actor, &actor_optimizer, &critic, &critic_optimizer, 0.99f);

	int success = 0;
	int fail = 0;

	tensor action({ _maze->ACTION_DIM() });

	test_agent(actor);

	for (int e = 0; e < p_episodes; e++)
	{
		float test_reward = 0;
		float train_reward = 0;
		_maze->reset();
		tensor state = _maze->get_state();

		while (!_maze->is_finished())
		{
			action.fill(0.f);
			action[random_generator::instance().choice(agent.get_action(&state).data(), _maze->ACTION_DIM())] = 1.f;
			//cout << action << endl;
			_maze->do_action(action);

			tensor next_state = _maze->get_state();
			const float reward = _maze->get_reward();
			const bool final = _maze->is_finished();

			agent.train(&state, &action, &next_state, reward, final);
			train_reward += reward;

			state = next_state;
		}

		_maze->reset();
		while (!_maze->is_finished())
		{
			tensor state = _maze->get_state();
			action.fill(0.f);
			action[random_generator::instance().choice(agent.get_action(&state).data(), _maze->ACTION_DIM())] = 1.f;
			_maze->do_action(action);
			test_reward += _maze->get_reward();
		}
		if (_maze->get_reward() == 1) success++; else fail++;

		cout << "Episode " << e << " " << success << " / " << fail << endl;
	}

	cout << _maze->to_string() << endl;
	test_agent(actor);
}

void maze_experiment::test_agent(neural_network& p_agent) const
{
	vector<char> actions = {'U','R','D','L'};
	tensor state({ static_cast<int>(_maze->mazeY() * _maze->mazeX()) });

	for(int i = 0; i < _maze->mazeY(); i++)
	{
		for (int j = 0; j < _maze->mazeX(); j++)
		{
			state.fill(0);
			state[i * _maze->mazeX() + j] = 1;
			cout << actions[p_agent.forward(&state).max_index()[0]];
		}
		cout << endl;
	}

	for (int i = 0; i < _maze->mazeY(); i++)
	{
		for (int j = 0; j < _maze->mazeX(); j++)
		{
			state.fill(0);
			state[i * _maze->mazeX() + j] = 1;
			cout << p_agent.forward(&state) << endl;
		}
	}
}
