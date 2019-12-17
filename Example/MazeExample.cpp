#include "MazeExample.h"
#include "RandomGenerator.h"
#include <iostream>
#include "QLearning.h"
#include "QuadraticCost.h"
#include "CoreLayer.h"
#include "RMSProp.h"
#include "SARSA.h"
#include "TD.h"
#include "PolicyGradient.h"
#include "DoubleQLearning.h"
#include "DeepQLearning.h"
#include "ICM.h"
#include "Encoder.h"
#include "ADAM.h"
#include "CountModule.h"
#include "Nadam.h"
#include "BackProph.h"
#include "BoltzmanExploration.h"
#include "EGreedyExploration.h"
#include "LinearInterpolation.h"
#include "ExponentialInterpolation.h"
#include "Logger.h"
#include "KLDivergence.h"
#include "CrossEntropyCost.h"
#include "PolicyGradient.h"
#include "NaturalGradient.h"
#include "NAC.h"
#include "A2C.h"
#include "ActorCritic.h"
#include "A3C.h"

using namespace Coeus;

MazeExample::MazeExample()
{
	_hConsole_c = CreateConsoleScreenBuffer(GENERIC_READ | GENERIC_WRITE, 0, NULL, CONSOLE_TEXTMODE_BUFFER, NULL);

	int topology[] = 
	{ 0, 0, 0, 0,
	  0, 2, 0, 2,
	  0, 0, 0, 2,
	  2, 0, 0, 0 };

	_maze = new Maze(topology, 4, 4, 15, false);
}


MazeExample::~MazeExample()
{
	delete _maze;
}

int MazeExample::example_q(int p_epochs, const bool p_verbose) {
	int hidden = 64;
	float limit = 0.01f;

	NeuralNetwork network;

	network.add_layer(new CoreLayer("hidden0", hidden, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit), _maze->STATE_DIM()));
	network.add_layer(new CoreLayer("hidden1", hidden / 2, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	network.add_layer(new CoreLayer("output", _maze->ACTION_DIM(), SIGMOID, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	// feed-forward connections
	network.add_connection("hidden0", "hidden1");
	network.add_connection("hidden1", "output");
	network.init();

	QLearning agent(&network, ADAM_RULE, 1e-3f, 0.99f);

	Tensor state0, state1;
	Tensor action({ _maze->ACTION_DIM() }, Tensor::ZERO);
	float reward = 0;
	const int epochs = p_epochs;

	EGreedyExploration exploration(0.9, new ExponentialInterpolation(0.9, 0.1, epochs));
	//BoltzmanExploration exploration(1, new ExponentialInterpolation(1, 0.1, epochs));

	int wins = 0, loses = 0;	
	SetConsoleActiveScreenBuffer(_hConsole_c);

	for (int e = 0; e < epochs; e++) {
		//if (p_verbose) cout << "Epoch " << e << endl;

		_maze->reset();

		state0 = _maze->get_state();

		while (!_maze->is_finished()) {
			//cout << maze->toString() << endl;
			network.activate(&state0);
			const int action0 = exploration.get_action(network.get_output());
			Encoder::one_hot(action, action0);
			_maze->do_action(action);

			state1 = _maze->get_state();
			reward = _maze->get_reward();

			agent.train(&state0, action0, &state1, reward, _maze->is_finished());
			state0.override(&state1);
		}

		exploration.update(e);

		if (_maze->is_winner()) {
			wins++;
		}
		else {
			loses++;
		}

		//cout << epsilon << endl;
		//cout << wins << " " << task.getEnvironment()->moves() << " " << reward << endl;

		//agent.reset_traces();
		if (p_verbose)
		{
			//cout << task.getEnvironment()->moves() << endl;
			//cout << wins << " / " << loses << endl;

			string s = "Q-Learning Episode " + to_string(e) + " results: " + to_string(wins) + " / " + to_string(loses);
			console_print(s, 0, 0);
		}
		if (e % 500 == 0) {
			//test_policy(network);
		}
	}
	
	test_policy(network);
	CloseHandle(_hConsole_c);
	return test_q(&network, true);
}

void MazeExample::example_double_q(int p_epochs, const bool p_verbose) {
	int hidden = 64;
	float limit = 0.01f;

	NeuralNetwork networkA;

	networkA.add_layer(new CoreLayer("hidden0", hidden, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit), _maze->STATE_DIM()));
	networkA.add_layer(new CoreLayer("hidden1", hidden / 2, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	networkA.add_layer(new CoreLayer("output", _maze->ACTION_DIM(), SIGMOID, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	// feed-forward connections
	networkA.add_connection("hidden0", "hidden1");
	networkA.add_connection("hidden1", "output");
	networkA.init();

	NeuralNetwork networkB;

	networkB.add_layer(new CoreLayer("hidden0", hidden, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit), _maze->STATE_DIM()));
	networkB.add_layer(new CoreLayer("hidden1", hidden / 2, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	networkB.add_layer(new CoreLayer("output", _maze->ACTION_DIM(), SIGMOID, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	// feed-forward connections
	networkB.add_connection("hidden0", "hidden1");
	networkB.add_connection("hidden1", "output");
	networkB.init();

	DoubleQLearning agent(&networkA, &networkB, ADAM_RULE, 1e-3f, 0.99f);

	Tensor state0, state1;
	Tensor action({ _maze->ACTION_DIM() }, Tensor::ZERO);
	float reward = 0;
	const int epochs = p_epochs;

	//EGreedyExploration exploration(0.9, new ExponentialInterpolation(0.9, 0.1, epochs));
	BoltzmanExploration exploration(1, new ExponentialInterpolation(1, 1, epochs));

	int wins = 0, loses = 0;
	SetConsoleActiveScreenBuffer(_hConsole_c);

	for (int e = 0; e < epochs; e++) {
		//if (p_verbose) cout << "Epoch " << e << endl;

		_maze->reset();

		state0 = _maze->get_state();

		while (!_maze->is_finished()) {
			//cout << maze->toString() << endl;
			const int action0 = exploration.get_action(agent.get_output(&state0));
			Encoder::one_hot(action, action0);
			_maze->do_action(action);

			state1 = _maze->get_state();
			reward = _maze->get_reward();

			agent.train(&state0, action0, &state1, reward, _maze->is_finished());
			state0.override(&state1);
		}

		exploration.update(e);

		if (_maze->is_winner()) {
			wins++;
		}
		else {
			loses++;
		}

		//cout << epsilon << endl;
		//cout << wins << " " << task.getEnvironment()->moves() << " " << reward << endl;

		//agent.reset_traces();
		if (p_verbose)
		{
			//cout << task.getEnvironment()->moves() << endl;
			//cout << wins << " / " << loses << endl;
			string s = "Double Q-Learning Episode " + to_string(e) + " results: " + to_string(wins) + " / " + to_string(loses);
			console_print(s, 0, 0);
		}
	}

	CloseHandle(_hConsole_c);

	int step = 0;
	
	_maze->reset();
	while (!_maze->is_finished() && step < 20) {
		state0 = _maze->get_state();
		const int action_index = choose_action(agent.get_output(&state0), 0);
		action.fill(0);
		action[action_index] = 1;
		_maze->do_action(action);
		step++;
	}

	//cout << task.getReward() << endl;

	if (p_verbose)
	{
		cout << _maze->toString() << endl;
		cout << _maze->moves() << endl;

		Tensor s = Tensor::Zero({ _maze->STATE_DIM() });

		for (unsigned int i = 0; i < _maze->mazeY(); i++)
		{
			for (unsigned int j = 0; j < _maze->mazeX(); j++)
			{
				const int a = i * _maze->mazeX() + j;
				Encoder::one_hot(s, a);
				//cout << s << endl;
				//cout << *p_network->get_output() << endl;

				switch (agent.get_output(&s)->max_value_index())
				{
				case 0:
					cout << "U";
					break;
				case 1:
					cout << "R";
					break;
				case 2:
					cout << "D";
					break;
				case 3:
					cout << "L";
					break;
				default:;
				}
			}

			cout << endl;
		}

		for (unsigned int i = 0; i < _maze->mazeY(); i++)
		{
			for (unsigned int j = 0; j < _maze->mazeX(); j++)
			{
				const int a = i * _maze->mazeX() + j;
				Encoder::one_hot(s, a);

				//cout << *agent.get_output(&s) << endl;
				for (int a = 0; a < agent.get_output(&s)->size(); a++) {
					printf("%1.2f ", (*agent.get_output(&s))[a]);
				}
				cout << endl;
			}
		}
	}
}

int MazeExample::example_sarsa(int p_epochs, const bool p_verbose) {
	int hidden = 64;
	float limit = 0.01f;

	NeuralNetwork network;

	network.add_layer(new CoreLayer("hidden0", hidden, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit), _maze->STATE_DIM()));
	network.add_layer(new CoreLayer("hidden1", hidden / 2, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	network.add_layer(new CoreLayer("output", _maze->ACTION_DIM(), SIGMOID, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	// feed-forward connections
	network.add_connection("hidden0", "hidden1");
	network.add_connection("hidden1", "output");
	network.init();

	SARSA agent(&network, ADAM_RULE, 1e-3f, 0.99f);

	Tensor state0, state1;
	Tensor action({ _maze->ACTION_DIM() }, Tensor::ZERO);
	float reward = 0;
	const int epochs = p_epochs;

	EGreedyExploration exploration(0.9, new LinearInterpolation(0.9, 0.1, epochs));
	//BoltzmanExploration exploration(1, new ExponentialInterpolation(1, 1, epochs));

	int wins = 0, loses = 0;
	SetConsoleActiveScreenBuffer(_hConsole_c);

	for (int e = 0; e < epochs; e++) {
		//if (p_verbose) cout << "Epoch " << e << endl;

		_maze->reset();

		state0 = _maze->get_state();
		network.activate(&state0);
		int action0 = exploration.get_action(network.get_output());

		while (!_maze->is_finished()) {
			//cout << maze->toString() << endl;
			Encoder::one_hot(action, action0);
			_maze->do_action(action);
			state1 = _maze->get_state();
			reward = _maze->get_reward();
			
			network.activate(&state1);
			const int action1 = exploration.get_action(network.get_output());

			agent.train(&state0, action0, &state1, action1, reward, _maze->is_finished());
			state0.override(&state1);
			action0 = action1;
		}

		exploration.update(e);

		if (_maze->is_winner()) {
			wins++;
		}
		else {
			loses++;
		}

		//cout << epsilon << endl;
		//cout << wins << " " << task.getEnvironment()->moves() << " " << reward << endl;

		//agent.reset_traces();
		if (p_verbose)
		{
			//cout << task.getEnvironment()->moves() << endl;
			//cout << wins << " / " << loses << endl;
			string s = "SARSA Episode " + to_string(e) + " results: " + to_string(wins) + " / " + to_string(loses);
			console_print(s, 0, 0);
		}
	}

	test_policy(network);
	CloseHandle(_hConsole_c);
	return test_q(&network, true);
}

void MazeExample::example_actor_critic(int p_epochs, bool p_verbose) {
	const int hidden = 32;
	const float limit = 0.01f;

	NeuralNetwork network_critic;

	network_critic.add_layer(new CoreLayer("hidden0", hidden, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit), _maze->STATE_DIM()));
	network_critic.add_layer(new CoreLayer("hidden1", hidden / 2, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	network_critic.add_layer(new CoreLayer("output", 1, SIGMOID, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	// feed-forward connections
	network_critic.add_connection("hidden0", "hidden1");
	network_critic.add_connection("hidden1", "output");
	network_critic.init();

	NeuralNetwork network_actor;

	network_actor.add_layer(new CoreLayer("hidden0", hidden, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit), _maze->STATE_DIM()));
	network_actor.add_layer(new CoreLayer("hidden1", hidden / 2, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	network_actor.add_layer(new CoreLayer("output", _maze->ACTION_DIM(), SOFTMAX, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	// feed-forward connections
	network_actor.add_connection("hidden0", "hidden1");
	network_actor.add_connection("hidden1", "output");
	network_actor.init();

	ActorCritic agent(&network_critic, ADAM_RULE, 1e-3f, 0.99f, &network_actor, ADAM_RULE, 1e-3f);

	Tensor state0, state1;
	Tensor action({ _maze->ACTION_DIM() }, Tensor::ZERO);
	float value0, value1;
	float reward = 0;
	int epochs = p_epochs;
	float td_error = 0;

	int wins = 0, loses = 0;
	SetConsoleActiveScreenBuffer(_hConsole_c);
	
	//Logger::instance().init("log.log");

	for (int e = 0; e < epochs; e++) {
		//cout << "Epoch " << e << endl;

		_maze->reset();
		state0 = _maze->get_state();

		network_critic.activate(&state0);

		while (!_maze->is_finished()) {
			network_actor.activate(&state0);

			const int action0 = RandomGenerator::get_instance().choice(network_actor.get_output()->arr(), _maze->ACTION_DIM());
			Encoder::one_hot(action, action0);
			_maze->do_action(action);
			state1 = _maze->get_state();
			reward = _maze->get_reward();
			agent.train(&state0, &action, &state1, reward, _maze->is_finished());

			state0.override(&state1);
		}

		//cout << task.getEnvironment()->moves() << endl;

		if (_maze->is_winner()) {
			wins++;
		}
		else {
			loses++;
		}

		string s = "Actor-Critic Episode " + to_string(e) + " results: " + to_string(wins) + " / " + to_string(loses);
		console_print(s, 0, 0);

		//Logger::instance().log(to_string(e) + ";" + to_string(cum_i_reward / task.getEnvironment()->moves()) + ";" + to_string(cum_e_reward / task.getEnvironment()->moves()));
		//cout << wins << " / " << loses << endl;
	}

	test_policy(network_actor);
	CloseHandle(_hConsole_c);

	//Logger::instance().close();

	test_q(&network_actor);
	cout << endl;
	test_v(&network_critic);
}

void MazeExample::example_nac(int p_epochs, bool p_verbose)
{
	int hidden = 32;
	float limit = 0.01f;

	NeuralNetwork network_critic;

	network_critic.add_layer(new CoreLayer("hidden0", hidden, RELU, new TensorInitializer(TensorInitializer::LECUN_UNIFORM), _maze->STATE_DIM()));
	network_critic.add_layer(new CoreLayer("hidden1", hidden / 2, RELU, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	network_critic.add_layer(new CoreLayer("output", 1, SIGMOID, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	// feed-forward connections
	network_critic.add_connection("hidden0", "hidden1");
	network_critic.add_connection("hidden1", "output");
	network_critic.init();

	NeuralNetwork network_actor;

	network_actor.add_layer(new CoreLayer("hidden0", hidden, RELU, new TensorInitializer(TensorInitializer::LECUN_UNIFORM), _maze->STATE_DIM()));
	network_actor.add_layer(new CoreLayer("hidden1", hidden / 2, RELU, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	network_actor.add_layer(new CoreLayer("output", _maze->ACTION_DIM(), SOFTMAX, new TensorInitializer(TensorInitializer::LECUN_UNIFORM)));
	// feed-forward connections
	network_actor.add_connection("hidden0", "hidden1");
	network_actor.add_connection("hidden1", "output");
	network_actor.init();

	NAC agent(&network_critic, ADAM_RULE, 1e-3f, 0.99f, 0.98f, &network_actor, 1e-1f);

	Tensor state0, state1;
	Tensor action({ _maze->ACTION_DIM() }, Tensor::ZERO);
	float value0, value1;
	float reward = 0;
	int epochs = p_epochs;
	float td_error = 0;

	int wins = 0, loses = 0;
	//SetConsoleActiveScreenBuffer(_hConsole_c);

	//Logger::instance().init("log.log");

	for (int e = 0; e < epochs; e++) {
		//cout << "Epoch " << e << endl;

		_maze->reset();
		state0 = _maze->get_state();

		while (!_maze->is_finished()) {
			network_actor.activate(&state0);
			cout << *network_actor.get_output() << endl;
			const int action0 = RandomGenerator::get_instance().choice(network_actor.get_output()->arr(), _maze->ACTION_DIM());
			Encoder::one_hot(action, action0);
			_maze->do_action(action);
			state1 = _maze->get_state();
			reward = _maze->get_reward();
			agent.add_sample(&state0, &action, &state1, reward, _maze->is_finished());

			state0.override(&state1);
		}

		agent.train();		

		//cout << task.getEnvironment()->moves() << endl;

		if (_maze->is_winner()) {
			wins++;
		}
		else {
			loses++;
		}

		string s = "NAC Episode " + to_string(e) + " results: " + to_string(wins) + " / " + to_string(loses);
		//console_print(s, 0, 0);
		cout << s << endl;

		//Logger::instance().log(to_string(e) + ";" + to_string(cum_i_reward / task.getEnvironment()->moves()) + ";" + to_string(cum_e_reward / task.getEnvironment()->moves()));

		//cout << wins << " / " << loses << endl;
	}

	SetConsoleActiveScreenBuffer(_hConsole_c);
	test_policy(network_actor);
	CloseHandle(_hConsole_c);

	//Logger::instance().close();

	test_q(&network_actor);
	cout << endl;
	test_v(&network_critic);
}

void MazeExample::example_deep_q(int p_epochs, const bool p_verbose) {
	int hidden = 64;
	float limit = 0.01f;

	NeuralNetwork network;

	network.add_layer(new CoreLayer("hidden0", hidden, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit), _maze->STATE_DIM()));
	network.add_layer(new CoreLayer("hidden1", hidden / 2, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	network.add_layer(new CoreLayer("output", _maze->ACTION_DIM(), SIGMOID, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	// feed-forward connections
	network.add_connection("hidden0", "hidden1");
	network.add_connection("hidden1", "output");
	network.init();

	DeepQLearning agent(&network, ADAM_RULE, 1e-3f, 0.99f, 1000, 64, 100);

	Tensor state0, state1;
	Tensor action({ _maze->ACTION_DIM() }, Tensor::ZERO);
	float reward = 0;
	const int epochs = p_epochs;

	EGreedyExploration exploration(0.9, new ExponentialInterpolation(0.9, 0.1, epochs));
	//BoltzmanExploration exploration(1, new ExponentialInterpolation(1, 0.1, epochs));

	int wins = 0, loses = 0;
	SetConsoleActiveScreenBuffer(_hConsole_c);

	for (int e = 0; e < epochs; e++) {
		//if (p_verbose) cout << "Epoch " << e << endl;

		_maze->reset();
		state0 = _maze->get_state();

		while (!_maze->is_finished()) {
			//cout << maze->toString() << endl;
			network.activate(&state0);
			const int action0 = exploration.get_action(network.get_output());
			Encoder::one_hot(action, action0);
			_maze->do_action(action);

			state1 = _maze->get_state();
			reward = _maze->get_reward();

			agent.train(&state0, action0, &state1, reward, _maze->is_finished());
			state0.override(&state1);
		}

		exploration.update(e);

		if (_maze->is_winner()) {
			wins++;
		}
		else {
			loses++;
		}

		//cout << epsilon << endl;
		//cout << wins << " " << task.getEnvironment()->moves() << " " << reward << endl;

		//agent.reset_traces();
		if (p_verbose)
		{
			//cout << task.getEnvironment()->moves() << endl;
			//cout << wins << " / " << loses << endl;

			string s = "Deep Q-Learning Episode " + to_string(e) + " results: " + to_string(wins) + " / " + to_string(loses);
			console_print(s, 0, 0);
		}
		if (e % 500 == 0) {
			//test_policy(network);
		}
	}

	test_policy(network);
	CloseHandle(_hConsole_c);
	test_q(&network, true);
}

void MazeExample::example_a2c(int p_epochs, bool p_verbose)
{
	const int env_size = 8;
	
	vector<IEnvironment*> maze_array;

	for(int i = 0; i < env_size; i++)
	{
		maze_array.push_back(new Maze(*_maze));
	}
	
	int hidden = 64;
	float limit = 0.01f;

	NeuralNetwork network_critic;

	network_critic.add_layer(new CoreLayer("hidden0", hidden, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit), _maze->STATE_DIM()));
	network_critic.add_layer(new CoreLayer("hidden1", hidden / 2, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	network_critic.add_layer(new CoreLayer("output", 1, SIGMOID, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	// feed-forward connections
	network_critic.add_connection("hidden0", "hidden1");
	network_critic.add_connection("hidden1", "output");
	network_critic.init();

	NeuralNetwork network_actor;

	network_actor.add_layer(new CoreLayer("hidden0", hidden, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit), _maze->STATE_DIM()));
	network_actor.add_layer(new CoreLayer("hidden1", hidden / 2, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	network_actor.add_layer(new CoreLayer("output", _maze->ACTION_DIM(), SOFTMAX, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	// feed-forward connections
	network_actor.add_connection("hidden0", "hidden1");
	network_actor.add_connection("hidden1", "output");
	network_actor.init();

	A2C agent(maze_array, &network_critic, ADAM_RULE, 1e-3f, 0.99f, &network_actor, ADAM_RULE, 5e-4f);

	Tensor state0, state1;
	Tensor action({ _maze->ACTION_DIM() }, Tensor::ZERO);
	float value0, value1;
	float reward = 0;
	int epochs = p_epochs;
	float td_error = 0;

	int wins = 0, loses = 0;
	SetConsoleActiveScreenBuffer(_hConsole_c);

	//Logger::instance().init("log.log");
	Tensor prob({ _maze->ACTION_DIM() }, Tensor::ZERO);

	for (int e = 0; e < epochs; e++) {
		//cout << "Epoch " << e << endl;
		agent.train(16);

		_maze->reset();
		state0 = _maze->get_state();

		while (!_maze->is_finished()) {
			network_actor.activate(&state0);

			const int action0 = network_actor.get_output()->max_value_index();
			Encoder::one_hot(action, action0);
			_maze->do_action(action);

			state1 = _maze->get_state();
			state0.override(&state1);
		}

		//cout << task.getEnvironment()->moves() << endl;

		if (_maze->is_winner()) {
			wins++;
		}
		else {
			loses++;
		}

		string s = "A2C Episode " + to_string(e) + " results: " + to_string(wins) + " / " + to_string(loses);
		console_print(s, 0, 0);

		//Logger::instance().log(to_string(e) + ";" + to_string(cum_i_reward / task.getEnvironment()->moves()) + ";" + to_string(cum_e_reward / task.getEnvironment()->moves()));
		//cout << wins << " / " << loses << endl;
	}

	test_policy(network_actor);
	CloseHandle(_hConsole_c);

	for (int i = 0; i < env_size; i++)
	{
		delete maze_array[i];
	}

	//Logger::instance().close();

	test_q(&network_actor);
	cout << endl;
	test_v(&network_critic);
}

void MazeExample::example_a3c(int p_epochs, bool p_verbose)
{
	const int env_size = 16;

	vector<IEnvironment*> maze_array;

	for (int i = 0; i < env_size; i++)
	{
		maze_array.push_back(new Maze(*_maze));
	}

	int hidden = 64;
	float limit = 0.01f;

	NeuralNetwork network_critic;

	network_critic.add_layer(new CoreLayer("hidden0", hidden, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit), _maze->STATE_DIM()));
	network_critic.add_layer(new CoreLayer("hidden1", hidden / 2, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	network_critic.add_layer(new CoreLayer("output", 1, SIGMOID, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	// feed-forward connections
	network_critic.add_connection("hidden0", "hidden1");
	network_critic.add_connection("hidden1", "output");
	network_critic.init();

	NeuralNetwork network_actor;

	network_actor.add_layer(new CoreLayer("hidden0", hidden, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit), _maze->STATE_DIM()));
	network_actor.add_layer(new CoreLayer("hidden1", hidden / 2, RELU, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	network_actor.add_layer(new CoreLayer("output", _maze->ACTION_DIM(), SOFTMAX, new TensorInitializer(TensorInitializer::UNIFORM, -limit, limit)));
	// feed-forward connections
	network_actor.add_connection("hidden0", "hidden1");
	network_actor.add_connection("hidden1", "output");
	network_actor.init();

	A3C agent(maze_array, &network_critic, RADAM_RULE, 1e-3f, 0.99f, &network_actor, RADAM_RULE, 9e-4f);

	Tensor state0, state1;
	Tensor action({ _maze->ACTION_DIM() }, Tensor::ZERO);
	float value0, value1;
	float reward = 0;
	int epochs = p_epochs;
	float td_error = 0;

	int wins = 0, loses = 0;
	SetConsoleActiveScreenBuffer(_hConsole_c);

	//Logger::instance().init("log.log");
	Tensor prob({ _maze->ACTION_DIM() }, Tensor::ZERO);

	for (int e = 0; e < epochs; e++) {
		//cout << "Epoch " << e << endl;
		agent.train(16, 2000);

		_maze->reset();
		state0 = _maze->get_state();

		while (!_maze->is_finished()) {
			network_actor.activate(&state0);

			const int action0 = network_actor.get_output()->max_value_index();
			Encoder::one_hot(action, action0);
			_maze->do_action(action);

			state1 = _maze->get_state();
			state0.override(&state1);
		}

		//cout << task.getEnvironment()->moves() << endl;

		if (_maze->is_winner()) {
			wins++;
		}
		else {
			loses++;
		}

		string s = "A3C Episode " + to_string(e) + " results: " + to_string(wins) + " / " + to_string(loses);
		console_print(s, 0, 0);

		//Logger::instance().log(to_string(e) + ";" + to_string(cum_i_reward / task.getEnvironment()->moves()) + ";" + to_string(cum_e_reward / task.getEnvironment()->moves()));
		//cout << wins << " / " << loses << endl;
	}

	test_policy(network_actor);
	CloseHandle(_hConsole_c);

	for (int i = 0; i < env_size; i++)
	{
		delete maze_array[i];
	}

	//Logger::instance().close();

	test_q(&network_actor);
	cout << endl;
	test_v(&network_critic);
}

void MazeExample::example_icm(int p_hidden) {

	NeuralNetwork network_critic;

	network_critic.add_layer(new CoreLayer("hidden0", p_hidden, SIGMOID, new TensorInitializer(TensorInitializer::UNIFORM, -0.1, 0.1), _maze->STATE_DIM()));
	network_critic.add_layer(new CoreLayer("hidden1", p_hidden / 2, SIGMOID, new TensorInitializer(TensorInitializer::UNIFORM, -0.1, 0.1)));
	network_critic.add_layer(new CoreLayer("output", 1, SIGMOID, new TensorInitializer(TensorInitializer::UNIFORM, -0.1, 0.1)));
	network_critic.add_connection("hidden0", "hidden1");
	network_critic.add_connection("hidden1", "output");
	network_critic.init();

	TD critic(&network_critic, ADAM_RULE, 1e-3f, 0.9);

	NeuralNetwork network_actor;

	network_actor.add_layer(new CoreLayer("hidden0", p_hidden, SIGMOID, new TensorInitializer(TensorInitializer::UNIFORM, -0.1, 0.1), _maze->STATE_DIM()));
	network_actor.add_layer(new CoreLayer("hidden1", p_hidden / 2, SIGMOID, new TensorInitializer(TensorInitializer::UNIFORM, -0.1, 0.1)));
	network_actor.add_layer(new CoreLayer("output", _maze->ACTION_DIM(), SOFTMAX, new TensorInitializer(TensorInitializer::UNIFORM, -0.1, 0.1)));
	network_actor.add_connection("hidden0", "hidden1");
	network_actor.add_connection("hidden1", "output");
	network_actor.init();

	//PolicyGradient actor(&network_actor, ADAM_RULE, 0.1f);

	NeuralNetwork network_model;

	network_model.add_layer(new CoreLayer("hidden0", p_hidden, SIGMOID, new TensorInitializer(TensorInitializer::UNIFORM, -0.1, 0.1), _maze->STATE_DIM() + _maze->ACTION_DIM()));
	network_model.add_layer(new CoreLayer("hidden1", p_hidden / 2, SIGMOID, new TensorInitializer(TensorInitializer::UNIFORM, -0.1, 0.1)));
	network_model.add_layer(new CoreLayer("output", _maze->STATE_DIM(), SIGMOID, new TensorInitializer(TensorInitializer::UNIFORM, -0.1, 0.1)));
	network_model.add_connection("hidden0", "hidden1");
	network_model.add_connection("hidden1", "output");
	network_model.init();

	ADAM optimizer_m(&network_model);
	optimizer_m.init(new QuadraticCost(), 1e-3f);
	ICM icm(&network_model, &optimizer_m);

	Tensor state0, state1;
	Tensor action({ _maze->ACTION_DIM() }, Tensor::ZERO);
	float value0, value1;
	float reward = 0;
	int epochs = 2000;
	float td_error = 0;

	int wins = 0, loses = 0;

	//EGreedyExploration exploration(1, new LinearInterpolation(1, 0.1, epochs));
	//BoltzmanExploration exploration(10, new LinearInterpolation(10, 0.1, epochs));
	//EGreedyExploration exploration(0);
	BoltzmanExploration exploration(1);

	Logger::instance().init("log.log");
	float cum_i_reward = 0;
	float cum_e_reward = 0;

	for (int e = 0; e < epochs; e++) {
		cout << "Epoch " << e << endl;

		_maze->reset();
		state0 = _maze->get_state();

		while (!_maze->is_finished()) {
			network_actor.activate(&state0);

			const int action0 = RandomGenerator::get_instance().choice(network_actor.get_output()->arr(), 4);
			Encoder::one_hot(action, action0);
			_maze->do_action(action);

			state1 = _maze->get_state();

			const float ir = icm.get_intrinsic_reward(&state0, &action, &state1, 1.f);

			cum_i_reward += ir;
			cum_e_reward += _maze->get_reward();

			reward = _maze->get_reward() + ir;
			//cout << reward << endl;
			td_error = critic.train(&state0, &state1, reward, _maze->is_finished());
			icm.train(&state0, &action, &state1);
			//icm.add(&state0, &action, &state1);
			//actor.train(&state0, action0, td_error);

			state0.override(&state1);
		}

		//icm.train(32);

		cout << _maze->moves() << endl;

		if (_maze->is_winner()) {
			wins++;
		}
		else {
			loses++;
		}

		Logger::instance().log(to_string(e) + ";" + to_string(cum_i_reward / _maze->moves()) + ";" + to_string(cum_e_reward / _maze->moves()));
		cum_i_reward = 0;
		cum_e_reward = 0;

		exploration.update(e);

		cout << wins << " / " << loses << endl;
	}

	/* ICM Test
	Tensor state = Tensor::Zero({ int(maze->mazeY() * maze->mazeX())});

	for (unsigned int i = 0; i < maze->mazeY(); i++)
	{
		for (unsigned int j = 0; j < maze->mazeX(); j++)
		{
			const int s = i * maze->mazeX() + j;
			Encoder::one_hot(state, s);

			for(int a = 0; a < 4; a++)
			{
				Encoder::one_hot(action, a);

				icm.activate(&state, &action);

				cout <<  state << ":" << a << " -> " << *icm.get_output() << endl;
			}
		}
	}
	*/

	Logger::instance().close();

	test_q(&network_actor);
	//test_v(&network_critic);
}

void MazeExample::example_selector(int p_hidden)
{
	NeuralNetwork network_critic;

	network_critic.add_layer(new CoreLayer("hidden0", p_hidden, SIGMOID, new TensorInitializer(TensorInitializer::UNIFORM, -0.1, 0.1), _maze->STATE_DIM()));
	network_critic.add_layer(new CoreLayer("hidden1", p_hidden / 2, SIGMOID, new TensorInitializer(TensorInitializer::UNIFORM, -0.1, 0.1)));
	network_critic.add_layer(new CoreLayer("output", 1, TANH, new TensorInitializer(TensorInitializer::UNIFORM, -0.1, 0.1)));
	// feed-forward connections
	network_critic.add_connection("hidden0", "hidden1");
	network_critic.add_connection("hidden1", "output");
	network_critic.init();

	TD critic(&network_critic, ADAM_RULE, 1e-3f, 0.9f);

	NeuralNetwork network_actor;

	network_actor.add_layer(new CoreLayer("hidden0", p_hidden, SIGMOID, new TensorInitializer(TensorInitializer::UNIFORM, -0.1, 0.1), _maze->STATE_DIM()));
	network_actor.add_layer(new CoreLayer("hidden1", p_hidden / 2, SIGMOID, new TensorInitializer(TensorInitializer::UNIFORM, -0.1, 0.1)));
	network_actor.add_layer(new CoreLayer("output", _maze->ACTION_DIM(), SOFTMAX, new TensorInitializer(TensorInitializer::UNIFORM, -0.1, 0.1)));
	// feed-forward connections
	network_actor.add_connection("hidden0", "hidden1");
	network_actor.add_connection("hidden1", "output");
	network_actor.init();

	//PolicyGradient actor(&network_actor, ADAM_RULE, 0.001f);

	NeuralNetwork network_predictor;

	network_predictor.add_layer(new CoreLayer("hidden0", p_hidden, SIGMOID, new TensorInitializer(TensorInitializer::UNIFORM, -0.1, 0.1), _maze->STATE_DIM()));
	network_predictor.add_layer(new CoreLayer("hidden1", p_hidden / 2, SIGMOID, new TensorInitializer(TensorInitializer::UNIFORM, -0.1, 0.1)));
	network_predictor.add_layer(new CoreLayer("output", 1, SIGMOID, new TensorInitializer(TensorInitializer::UNIFORM, -0.1, 0.1)));
	// feed-forward connections
	network_predictor.add_connection("hidden0", "hidden1");
	network_predictor.add_connection("hidden1", "output");
	network_predictor.init();

	ADAM predictor(&network_predictor);
	predictor.init(new QuadraticCost(), 1e-3f);

	NeuralNetwork network_selector;

	network_selector.add_layer(new CoreLayer("hidden0", p_hidden, SIGMOID, new TensorInitializer(TensorInitializer::UNIFORM, -0.1, 0.1), _maze->STATE_DIM()));
	network_selector.add_layer(new CoreLayer("hidden1", p_hidden / 2, SIGMOID, new TensorInitializer(TensorInitializer::UNIFORM, -0.1, 0.1)));
	network_selector.add_layer(new CoreLayer("output", 2, SOFTMAX, new TensorInitializer(TensorInitializer::UNIFORM, -0.1, 0.1)));
	// feed-forward connections
	network_selector.add_connection("hidden0", "hidden1");
	network_selector.add_connection("hidden1", "output");
	network_selector.init();

	//PolicyGradient selector(&network_selector, ADAM_RULE, 0.01f);

	Tensor state0, state1;
	Tensor action({ _maze->ACTION_DIM() }, Tensor::ZERO);
	float value0, value1;
	int epochs = 1000;
	float td_error = 0;

	int wins = 0, loses = 0;

	CountModule count_module(_maze->STATE_DIM());

	Logger::instance().init("log.log");

	Tensor goal = Tensor::Zero({ _maze->STATE_DIM() });
	goal[_maze->goal()] = 1;
	Tensor predictor_target = Tensor::Zero({ 1 });
	bool goal_reached = false;
	int selection = 0;

	for (int e = 0; e < epochs; e++) {
		cout << "Epoch " << e << endl;

		float cum_i_reward = 0;
		float cum_e_reward = 0;
		_maze->reset();
		state0 = _maze->get_state();

		network_selector.activate(&goal);		

		if (e % 25 == 0) {
			selection = RandomGenerator::get_instance().choice(network_selector.get_output()->arr(), 2);
			//goal.fill(0);
			//goal[RandomGenerator::get_instance().random(0, env_dim - 1)] = 1;
		}

		while (!_maze->is_finished()) {
			network_actor.activate(&state0);

			const int action0 = RandomGenerator::get_instance().choice(network_actor.get_output()->arr(), 4);
			//cout << action0 << endl;
			Encoder::one_hot(action, action0);
			_maze->do_action(action);

			state1 = _maze->get_state();
			count_module.update(&state1);
			
			float ir = 0;

			if (selection == 0)
			{
				ir = count_module.uncertainty_motivation();
			}
			else
			{
				ir = count_module.familiarity_motivation();
				//cout << ir << endl;
			}

			const float r = _maze->get_reward() + ir;

			cum_i_reward += ir;
			cum_e_reward += _maze->get_reward();

			td_error = critic.train(&state0, &state1, r, _maze->is_finished());
			//actor.train(&state0, action0, td_error);

			state0.override(&state1);
			goal_reached = state0.max_value_index() == goal.max_value_index();
		}

		Logger::instance().log(
			to_string(e) + ";" + 
			to_string(cum_i_reward / _maze->moves()) + ";" +
			to_string(cum_e_reward / _maze->moves()) + ";" +
			to_string(network_selector.get_output()->at(0)) + ";" +
			to_string(network_selector.get_output()->at(1)) + ";"
		);

		network_predictor.activate(&goal);

		const float delta = goal_reached ? 1 : 0 - network_predictor.get_output()->at(0);
		predictor_target[0] = goal_reached ? 1 : 0;

		//selector.train(&goal, selection, delta);
		predictor.train(&goal, &predictor_target);

		cout << goal_reached << " " << goal.max_value_index() << " " << selection << " " << *network_selector.get_output() << endl;
		cout << _maze->moves() << endl;


		if (_maze->is_winner()) {
			wins++;
		}
		else {
			loses++;
		}

		cout << wins << " / " << loses << endl;
	}

	Logger::instance().close();

	test_q(&network_actor);
}

int MazeExample::test_q(NeuralNetwork* p_network, const bool p_verbose) const
{
	_maze->reset();

	Tensor action = Tensor::Zero({ _maze->ACTION_DIM() });
	Tensor state;

	int step = 0;
	float epsilon = 0;

	//cout << "--- TEST ---" << endl;
	/*
	while (!task.isFinished() && step < 20) {
		sensors = maze->getSensors();
		state = encode_state(&sensors);
		p_network->activate(&state);
		const int action_index = choose_action(p_network->get_output(), epsilon);
		Encoder::one_hot(action, action_index);
		maze->performAction(action_index);
		step++;		
	}
	*/
	//cout << task.getReward() << endl;

	if (p_verbose)
	{
		cout << endl;

		char action_label[4] = { 'U','R','D','L' };
		cout << _maze->toString() << endl;
		//cout << task.getEnvironment()->moves() << endl;

		Tensor s = Tensor::Zero({ _maze->STATE_DIM() });

		cout << endl;

		for (unsigned int i = 0; i < _maze->mazeY(); i++)
		{
			for (unsigned int j = 0; j < _maze->mazeX(); j++)
			{
				const int a = i * _maze->mazeX() + j;
				Encoder::one_hot(s, a);

				p_network->activate(&s);
				//cout << s << endl;
				//cout << *p_network->get_output() << endl;
				cout << action_label[p_network->get_output()->max_value_index()];
			}

			cout << endl;
		}

		cout << endl;

		for (unsigned int i = 0; i < _maze->mazeY(); i++)
		{
			for (unsigned int j = 0; j < _maze->mazeX(); j++)
			{
				const int a = i * _maze->mazeX() + j;
				Encoder::one_hot(s, a);

				p_network->activate(&s);
				//cout << *p_network->get_output() << endl;
				for (int a = 0; a < p_network->get_output()->size(); a++) {
					printf("%1.4f ", (*p_network->get_output())[a]);
				}
				cout << endl;
			}
		}
	}

	int result = 0;

	if (_maze->is_winner())
	{
		result = 1;
	}

	return result;
}

void MazeExample::test_v(NeuralNetwork* p_network, bool p_verbose) const
{
	_maze->reset();

	vector<float> sensors;
	Tensor action = Tensor::Zero({ _maze->ACTION_DIM() });
	Tensor state;

	Tensor s = Tensor::Zero({ _maze->STATE_DIM() });

	for (unsigned int i = 0; i < _maze->mazeY(); i++)
	{
		for (unsigned int j = 0; j < _maze->mazeX(); j++)
		{
			const int a = i * _maze->mazeX() + j;
			Encoder::one_hot(s, a);

			p_network->activate(&s);

			//cout << p_network->get_output()[0] << " ";
			printf("%1.2f ", (*p_network->get_output())[0]);
		}

		cout << endl;
	}
}

void MazeExample::test_policy(NeuralNetwork &p_network)
{
	_maze->reset();

	vector<float> sensors;
	Tensor action = Tensor::Zero({ _maze->ACTION_DIM() });
	Tensor state;

	string action_labels[4] = { "Up","Right","Down","Left" };
	int step = 0;
	float epsilon = 0;

	//cout << "--- TEST ---" << endl;
	console_clear();

	while (!_maze->is_finished() && step < 10) {
		for (int i = 0; i < _maze->mazeY(); i++) {
			console_print(_maze->toString(i), 0, i + 1);
		}


		state = _maze->get_state();
		p_network.activate(&state);
		string s;
		for (int i = 0; i < p_network.get_output()->size(); i++) {
			s += string_format("%s: %1.4f ", action_labels[i].c_str(), (*p_network.get_output())[i]);
		}
		s += " -> Step " + to_string(step) + " (" + action_labels[p_network.get_output()->max_value_index()] + ")";

		console_print(s, 0, 0);
		console_wait();

		const int action_index = choose_action(p_network.get_output(), epsilon);
		Encoder::one_hot(action, action_index);
		_maze->do_action(action);
		step++;	
	}

	/*
	for (int i = 0; i < maze->mazeY(); i++) {
		console_print(maze->toString(i), 0, i + 1);
	}
	console_wait();
	*/
	console_clear();
}

Tensor MazeExample::encode_state(vector<float>* p_sensors) {
	const Tensor res({ static_cast<int>(p_sensors->size()) }, Tensor::ZERO);

	for (unsigned int i = 0; i < p_sensors->size(); i++) {
		res[i] = p_sensors->at(i);
	}

	return Tensor(res);
}

int MazeExample::choose_action(Tensor* p_input, const float epsilon) {
	int action = 0;
	const float random = RandomGenerator::get_instance().random();

	//cout << *p_input << endl;

	if (random < epsilon) {
		action = RandomGenerator::get_instance().random(0, 3);
	}
	else {
		/*
		for (int i = 0; i < p_input->size(); i++) {
			if ((*p_input)[i] != (*p_input)[i])
			{
				assert(0);
			}
			if ((*p_input)[i] >(*p_input)[action]) {
				action = i;
			}
		}
		*/
		action = p_input->max_value_index();
	}

	return action;
}

void MazeExample::binary_encoding(const int p_value, Tensor* p_vector) {
	p_vector->fill(0);
	(*p_vector)[p_value] = 1.f;
}

void MazeExample::console_print(string & p_s, int p_x, int p_y)
{
	wstring stemp = wstring(p_s.begin(), p_s.end());
	LPCWSTR str = stemp.c_str();
	DWORD len = wcslen(str);
	DWORD dwBytesWritten = 0;
	COORD pos = { p_x, p_y };
	WriteConsoleOutputCharacter(_hConsole_c, str, len, pos, &dwBytesWritten);
}

void MazeExample::console_clear()
{
	WCHAR fill = ' ';
	COORD pos = { 0, 0 };
	CONSOLE_SCREEN_BUFFER_INFO s;
	GetConsoleScreenBufferInfo(_hConsole_c, &s);
	DWORD written, cells = s.dwSize.X * s.dwSize.Y;
	FillConsoleOutputCharacter(_hConsole_c, fill, cells, pos, &written);
	FillConsoleOutputAttribute(_hConsole_c, s.wAttributes, cells, pos, &written);
}

TCHAR MazeExample::console_wait()
{
	TCHAR  ch;
	DWORD  mode;
	DWORD  count;
	HANDLE hstdin = GetStdHandle(STD_INPUT_HANDLE);

	// Switch to raw mode
	GetConsoleMode(hstdin, &mode);
	SetConsoleMode(hstdin, 0);

	// Wait for the user's response
	WaitForSingleObject(hstdin, INFINITE);

	// Read the (single) key pressed
	ReadConsole(hstdin, &ch, 1, &count, NULL);

	// Restore the console to its previous state
	SetConsoleMode(hstdin, mode);

	// Return the key code
	return ch;
}

string MazeExample::string_format(const string fmt_str, ...) {
	int final_n, n = ((int)fmt_str.size()) * 2; /* Reserve two times as much as the length of the fmt_str */
	unique_ptr<char[]> formatted;
	va_list ap;
	while (1) {
		formatted.reset(new char[n]); /* Wrap the plain char array into the unique_ptr */
		strcpy(&formatted[0], fmt_str.c_str());
		va_start(ap, fmt_str);
		final_n = vsnprintf(&formatted[0], n, fmt_str.c_str(), ap);
		va_end(ap);
		if (final_n < 0 || final_n >= n)
			n += abs(final_n - n + 1);
		else
			break;
	}
	return string(formatted.get());
}
