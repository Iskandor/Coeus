#include "MazeExample.h"
#include "RandomGenerator.h"
#include <iostream>
#include "QLearning.h"
#include "MazeTask.h"
#include "QuadraticCost.h"
#include "CoreLayer.h"
#include "RMSProp.h"
#include "SARSA.h"
#include "TD.h"
#include "Actor.h"
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

using namespace Coeus;

MazeExample::MazeExample()
{
}


MazeExample::~MazeExample()
{
}

int MazeExample::example_q(int p_hidden, float p_alpha, float p_lambda,  const bool p_verbose) {
	MazeTask task;
	Maze* maze = task.getEnvironment();

	NeuralNetwork network;

	network.add_layer(new CoreLayer("hidden0", p_hidden, TANH, new TensorInitializer(UNIFORM, -0.1, 0.1), 25));
	network.add_layer(new CoreLayer("hidden1", p_hidden / 2, TANH, new TensorInitializer(UNIFORM, -0.1, 0.1)));
	network.add_layer(new CoreLayer("output", 4, TANH, new TensorInitializer(UNIFORM, -0.1, 0.1)));
	// feed-forward connections
	network.add_connection("hidden0", "hidden1");
	network.add_connection("hidden1", "output");
	network.init();

	//BackProp optimizer(&network);
	//optimizer.init(new QuadraticCost(), p_alpha, 0.9, true);
	ADAM optimizer(&network);
	//RMSProp optimizer(&network);
	optimizer.init(new QuadraticCost(), p_alpha);
	//optimizer.add_learning_rate_module(new WarmStartup(1e-3, 1e-2, 10, 2));
	//CountModule curiosity(maze->mazeX() * maze->mazeY());
	//QLearning agent(&network, ADAM_RULE, p_alpha, 0.9, p_lambda);
	QLearning agent(&network, &optimizer, 0.9f);

	vector<float> sensors;
	Tensor state0, state1;
	float reward = 0;
	const int epochs = 10000;

	EGreedyExploration exploration(0.3, new ExponentialInterpolation(0.3, 0.1, epochs));
	//BoltzmanExploration exploration(10, new ExponentialInterpolation(10, 0.1, epochs));

	int wins = 0, loses = 0;

	for (int e = 0; e < epochs; e++) {
		//if (p_verbose) cout << "Epoch " << e << endl;

		task.getEnvironment()->reset();

		sensors = maze->getSensors();
		state0 = encode_state(&sensors);

		while (!task.isFinished()) {
			//cout << maze->toString() << endl;
			network.activate(&state0);
			const int action0 = exploration.get_action(network.get_output());
			maze->performAction(action0);

			sensors = maze->getSensors();
			state1 = encode_state(&sensors);
			//curiosity.update(&state1);
			reward = task.getReward(); // +curiosity.get_reward(&state1);

			agent.train(&state0, action0, &state1, reward);
			state0.override(&state1);
		}

		exploration.update(e);

		if (task.isWinner()) {
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
			cout << task.getEnvironment()->moves() << endl;
			cout << wins << " / " << loses << endl;
		}
	}

	return test_q(&network, true);
}

void MazeExample::example_double_q() {
	MazeTask task;
	Maze* maze = task.getEnvironment();

	NeuralNetwork network_a;

	network_a.add_layer(new CoreLayer("hidden0", 164, RELU, new TensorInitializer(LECUN_UNIFORM), 64));
	network_a.add_layer(new CoreLayer("output", 4, LINEAR, new TensorInitializer(LECUN_UNIFORM)));
	// feed-forward connections
	network_a.add_connection("hidden0", "output");
	network_a.init();

	RMSProp optimizer1(&network_a);
	optimizer1.init(new QuadraticCost(), 0.003f);

	NeuralNetwork network_b;

	network_b.add_layer(new CoreLayer("hidden0", 164, RELU, new TensorInitializer(LECUN_UNIFORM), 64));
	network_b.add_layer(new CoreLayer("output", 4, LINEAR, new TensorInitializer(LECUN_UNIFORM)));
	// feed-forward connections
	network_b.add_connection("hidden0", "output");
	network_b.init();

	RMSProp optimizer2(&network_b);
	optimizer2.init(new QuadraticCost(), 0.003f);

	DoubleQLearning critic(&network_a, &optimizer1, &network_b, &optimizer2, 0.9f);

	vector<float> sensors;
	Tensor state0, state1;
	Tensor output;
	int action0;
	float reward = 0;
	float epsilon = 1;
	int epochs = 5000;

	int wins = 0, loses = 0;

	//FILE* pFile = fopen("application.log", "w");
	//Output2FILE::Stream() = pFile;
	//FILELog::ReportingLevel() = FILELog::FromString("DEBUG1");

	for (int e = 0; e < epochs; e++) {
		cout << "Epoch " << e << endl;

		task.getEnvironment()->reset();

		while (!task.isFinished()) {
			//cout << maze->toString() << endl;

			sensors = maze->getSensors();
			state0 = encode_state(&sensors);
			network_a.activate(&state0);
			network_b.activate(&state0);

			//output = (*network_a.get_output() + *network_b.get_output()) / 2;
			action0 = choose_action(&output, epsilon);
			maze->performAction(action0);

			sensors = maze->getSensors();
			state1 = encode_state(&sensors);
			reward = task.getReward();

			critic.train(&state0, action0, &state1, reward);
		}

		cout << task.getEnvironment()->moves() << endl;
		cout << epsilon << endl;

		if (reward > 0) {
			wins++;
		}
		else {
			loses++;
		}

		//cout << maze->toString() << endl;
		cout << wins << " / " << loses << endl;
		//FILE_LOG(logDEBUG1) << wins << " " << loses;


		//exploration->update((float)e / epochs);

		if (epsilon > 0.1) {
			epsilon -= (1.0f / epochs);
		}
	}
}

int MazeExample::example_sarsa(int p_hidden, float p_alpha, float p_lambda, const bool p_verbose) {
	MazeTask task;
	Maze* maze = task.getEnvironment();

	NeuralNetwork network;

	network.add_layer(new CoreLayer("hidden0", p_hidden, TANH, new TensorInitializer(UNIFORM, -0.1, 0.1), 25));
	network.add_layer(new CoreLayer("hidden1", p_hidden / 2, TANH, new TensorInitializer(UNIFORM, -0.1, 0.1)));
	network.add_layer(new CoreLayer("output", 4, TANH, new TensorInitializer(UNIFORM, -0.1, 0.1)));
	// feed-forward connections
	network.add_connection("hidden0", "hidden1");
	network.add_connection("hidden1", "output");
	network.init();

	//BackProp optimizer(&network);
	//optimizer.init(new QuadraticCost(), p_alpha, 0.9, true);
	ADAM optimizer(&network);
	//RMSProp optimizer(&network);
	optimizer.init(new QuadraticCost(), p_alpha);
	//optimizer.add_learning_rate_module(new WarmStartup(1e-3, 1e-2, 10, 2));
	//CountModule curiosity(maze->mazeX() * maze->mazeY());
	//QLearning agent(&network, ADAM_RULE, p_alpha, 0.9, p_lambda);
	SARSA agent(&network, &optimizer, 0.9f);

	vector<float> sensors;
	Tensor state0, state1;
	float reward = 0;
	const int epochs = 10000;

	//EGreedyExploration exploration(1, new LinearInterpolation(1, 0.1, epochs));
	BoltzmanExploration exploration(10, new LinearInterpolation(10, 0.1, epochs));

	int wins = 0, loses = 0;

	for (int e = 0; e < epochs; e++) {
		//if (p_verbose) cout << "Epoch " << e << endl;

		task.getEnvironment()->reset();

		sensors = maze->getSensors();
		state0 = encode_state(&sensors);
		network.activate(&state0);
		int action0 = exploration.get_action(network.get_output());

		while (!task.isFinished()) {
			//cout << maze->toString() << endl;
			maze->performAction(action0);
			sensors = maze->getSensors();
			reward = task.getReward();

			state1 = encode_state(&sensors);
			network.activate(&state1);
			const int action1 = exploration.get_action(network.get_output());

			agent.train(&state0, action0, &state1, action1, reward, task.isFinished());
			state0.override(&state1);
			action0 = action1;
		}

		exploration.update(e);

		if (task.isWinner()) {
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
			cout << task.getEnvironment()->moves() << endl;
			cout << wins << " / " << loses << endl;
		}
	}

	return test_q(&network, true);
}

void MazeExample::example_actor_critic(int p_hidden) {
	MazeTask task;
	Maze* maze = task.getEnvironment();

	NeuralNetwork network_critic;

	network_critic.add_layer(new CoreLayer("hidden0", p_hidden, SIGMOID, new TensorInitializer(UNIFORM, -0.1, 0.1), 25));
	network_critic.add_layer(new CoreLayer("hidden1", p_hidden / 2, SIGMOID, new TensorInitializer(UNIFORM, -0.1, 0.1)));
	network_critic.add_layer(new CoreLayer("output", 1, SIGMOID, new TensorInitializer(UNIFORM, -0.1, 0.1)));
	// feed-forward connections
	network_critic.add_connection("hidden0", "hidden1");
	network_critic.add_connection("hidden1", "output");
	network_critic.init();

	//TD critic(&network_critic, ADAM_RULE, 2e-4f, 0.9f);
	ADAM optimizer_c(&network_critic);
	optimizer_c.init(new QuadraticCost(), 1e-3f);
	TD critic(&network_critic, &optimizer_c, 0.9);

	NeuralNetwork network_actor;

	network_actor.add_layer(new CoreLayer("hidden0", p_hidden, SIGMOID, new TensorInitializer(UNIFORM, -0.1, 0.1), 25));
	network_actor.add_layer(new CoreLayer("hidden1", p_hidden / 2, SIGMOID, new TensorInitializer(UNIFORM, -0.1, 0.1)));
	network_actor.add_layer(new CoreLayer("output", 4, SIGMOID, new TensorInitializer(UNIFORM, -0.1, 0.1)));
	// feed-forward connections
	network_actor.add_connection("hidden0", "hidden1");
	network_actor.add_connection("hidden1", "output");
	network_actor.init();

	//Actor actor(&network_actor, ADAM_RULE, 1e-4);
	ADAM optimizer_a(&network_actor);
	optimizer_a.init(new QuadraticCost(), 1e-3f);
	Actor actor(&network_actor, &optimizer_a, 0.01f);

	vector<float> sensors;
	Tensor state0, state1;
	int action0;
	float value0, value1;
	float reward = 0;
	int epochs = 10000;
	float td_error = 0;

	int wins = 0, loses = 0;

	//EGreedyExploration exploration(1, new LinearInterpolation(1, 0.1, epochs));
	//BoltzmanExploration exploration(10, new LinearInterpolation(10, 0.1, epochs));
	BoltzmanExploration exploration(1);

	CountModule count_module(25);

	Logger::instance().init("log.log");
	float cum_reward = 0;

	for (int e = 0; e < epochs; e++) {
		cout << "Epoch " << e << endl;

		task.getEnvironment()->reset();
		sensors = maze->getSensors();
		state0 = encode_state(&sensors);

		while (!task.isFinished()) {
			network_actor.activate(&state0);

			action0 = exploration.get_action(network_actor.get_output());
			maze->performAction(action0);

			sensors = maze->getSensors();
			state1 = encode_state(&sensors);
			count_module.update(&state1);

			cum_reward += count_module.get_reward(&state1);

			reward = task.getReward() + count_module.get_reward(&state1);
			td_error = critic.train(&state0, &state1, reward);
			
			actor.train(&state0, action0, td_error);

			state0.override(&state1);
		}

		cout << task.getEnvironment()->moves() << endl;

		if (task.isWinner()) {
			wins++;
		}
		else {
			loses++;
		}

		exploration.update(e);

		if ((e + 1) % 10 == 0)
		{
			Logger::instance().log(to_string(cum_reward / 10));
			cum_reward = 0;
		}

		cout << wins << " / " << loses << endl;
	}

	Logger::instance().close();

	test_q(&network_actor);
}

int MazeExample::example_deep_q(int p_hidden, float p_alpha, float p_lambda, const bool p_verbose) {
	MazeTask task;
	Maze* maze = task.getEnvironment();

	NeuralNetwork network;

	network.add_layer(new CoreLayer("hidden0", p_hidden, TANH, new TensorInitializer(UNIFORM, -0.1, 0.1), 25));
	network.add_layer(new CoreLayer("hidden1", p_hidden / 2, TANH, new TensorInitializer(UNIFORM, -0.1, 0.1)));
	network.add_layer(new CoreLayer("output", 4, TANH, new TensorInitializer(UNIFORM, -0.1, 0.1)));
	// feed-forward connections
	network.add_connection("hidden0", "hidden1");
	network.add_connection("hidden1", "output");
	network.init();

	//BackProp optimizer(&network);
	//optimizer.init(new QuadraticCost(), p_alpha, 0.9, true);
	ADAM optimizer(&network);
	//RMSProp optimizer(&network);
	optimizer.init(new QuadraticCost(), p_alpha);
	//optimizer.add_learning_rate_module(new WarmStartup(1e-3, 1e-2, 10, 2));
	//CountModule curiosity(maze->mazeX() * maze->mazeY());
	//QLearning agent(&network, ADAM_RULE, p_alpha, 0.9, p_lambda);
	DeepQLearning agent(&network, &optimizer, 0.9f, 128, 32);

	vector<float> sensors;
	Tensor state0, state1;
	float reward = 0;
	const int epochs = 10000;

	EGreedyExploration exploration(0.3, new ExponentialInterpolation(0.3, 0.1, epochs));
	//BoltzmanExploration exploration(10, new ExponentialInterpolation(10, 0.1, epochs));

	int wins = 0, loses = 0;

	for (int e = 0; e < epochs; e++) {
		//if (p_verbose) cout << "Epoch " << e << endl;

		task.getEnvironment()->reset();

		sensors = maze->getSensors();
		state0 = encode_state(&sensors);

		while (!task.isFinished()) {
			//cout << maze->toString() << endl;
			network.activate(&state0);
			const int action0 = exploration.get_action(network.get_output());
			maze->performAction(action0);

			sensors = maze->getSensors();
			state1 = encode_state(&sensors);
			//curiosity.update(&state1);
			reward = task.getReward(); // +curiosity.get_reward(&state1);

			agent.train(&state0, action0, &state1, reward, task.isFinished());
			state0.override(&state1);
		}

		exploration.update(e);

		if (task.isWinner()) {
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
			cout << task.getEnvironment()->moves() << endl;
			cout << wins << " / " << loses << endl;
		}
	}

	return test_q(&network, true);
}

void MazeExample::example_icm(int p_hidden) {
	MazeTask task;
	Maze* maze = task.getEnvironment();

	NeuralNetwork network_critic;

	network_critic.add_layer(new CoreLayer("hidden0", p_hidden, SIGMOID, new TensorInitializer(UNIFORM, -0.1, 0.1), maze->mazeY() * maze->mazeX()));
	network_critic.add_layer(new CoreLayer("hidden1", p_hidden / 2, SIGMOID, new TensorInitializer(UNIFORM, -0.1, 0.1)));
	network_critic.add_layer(new CoreLayer("output", 1, SIGMOID, new TensorInitializer(UNIFORM, -0.1, 0.1)));
	network_critic.add_connection("hidden0", "hidden1");
	network_critic.add_connection("hidden1", "output");
	network_critic.init();

	ADAM optimizer_c(&network_critic);
	optimizer_c.init(new QuadraticCost(), 1e-3f);
	TD critic(&network_critic, &optimizer_c, 0.9);

	NeuralNetwork network_actor;

	network_actor.add_layer(new CoreLayer("hidden0", p_hidden, SIGMOID, new TensorInitializer(UNIFORM, -0.1, 0.1), maze->mazeY() * maze->mazeX()));
	network_actor.add_layer(new CoreLayer("hidden1", p_hidden / 2, SIGMOID, new TensorInitializer(UNIFORM, -0.1, 0.1)));
	network_actor.add_layer(new CoreLayer("output", 4, SIGMOID, new TensorInitializer(UNIFORM, -0.1, 0.1)));
	network_actor.add_connection("hidden0", "hidden1");
	network_actor.add_connection("hidden1", "output");
	network_actor.init();

	ADAM optimizer_a(&network_actor);
	optimizer_a.init(new QuadraticCost(), 1e-3f);
	Actor actor(&network_actor, &optimizer_a, 0.01f);

	NeuralNetwork network_model;

	network_model.add_layer(new CoreLayer("hidden0", p_hidden, SIGMOID, new TensorInitializer(UNIFORM, -0.1, 0.1), maze->mazeY() * maze->mazeX() + 4));
	network_model.add_layer(new CoreLayer("hidden1", p_hidden / 2, SIGMOID, new TensorInitializer(UNIFORM, -0.1, 0.1)));
	network_model.add_layer(new CoreLayer("output", 25, SIGMOID, new TensorInitializer(UNIFORM, -0.1, 0.1)));
	network_model.add_connection("hidden0", "hidden1");
	network_model.add_connection("hidden1", "output");
	network_model.init();

	ADAM optimizer_m(&network_model);
	optimizer_m.init(new QuadraticCost(), 1e-3f);
	ICM icm(&network_model, &optimizer_m, 128);

	vector<float> sensors;
	Tensor state0, state1, action;
	float value0, value1;
	float reward = 0;
	int epochs = 15000;
	float td_error = 0;

	int wins = 0, loses = 0;

	action = Tensor::Zero({ 4 });

	//EGreedyExploration exploration(1, new LinearInterpolation(1, 0.1, epochs));
	//BoltzmanExploration exploration(10, new LinearInterpolation(10, 0.1, epochs));
	//EGreedyExploration exploration(0);
	BoltzmanExploration exploration(1);

	Logger::instance().init("log.log");
	float cum_reward = 0;

	for (int e = 0; e < epochs; e++) {
		cout << "Epoch " << e << endl;

		task.getEnvironment()->reset();
		sensors = maze->getSensors();
		state0 = encode_state(&sensors);

		while (!task.isFinished()) {
			network_actor.activate(&state0);

			const int action0 = exploration.get_action(network_actor.get_output());
			Encoder::one_hot(action, action0);
			maze->performAction(action0);

			sensors = maze->getSensors();
			state1 = encode_state(&sensors);

			//const float ir = icm.get_intrinsic_reward(&state0, &action, &state1, 1.f/25);

			//cum_reward += ir;

			reward = task.getReward() + 0;
			//cout << reward << endl;
			td_error = critic.train(&state0, &state1, reward);
			//icm.add(&state0, &action, &state1);
			actor.train(&state0, action0, td_error);

			state0.override(&state1);
		}

		//icm.train(32);

		cout << task.getEnvironment()->moves() << endl;

		if (task.isWinner()) {
			wins++;
		}
		else {
			loses++;
		}

		if ((e+1) % 10 == 0)
		{
			Logger::instance().log(to_string(cum_reward / 10));
			cum_reward = 0;
		}

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

int MazeExample::test_q(NeuralNetwork* p_network, const bool p_verbose) const
{
	MazeTask task;
	Maze* maze = task.getEnvironment();
	task.getEnvironment()->reset();

	vector<float> sensors;
	Tensor action = Tensor::Zero({ 4 });
	Tensor state;

	int step = 0;
	float epsilon = 0;

	//cout << "--- TEST ---" << endl;

	while (!task.isFinished() && step < 20) {
		sensors = maze->getSensors();
		state = encode_state(&sensors);
		p_network->activate(&state);
		const int action_index = choose_action(p_network->get_output(), epsilon);
		Encoder::one_hot(action, action_index);
		maze->performAction(action_index);
		step++;		
	}

	//cout << task.getReward() << endl;

	if (p_verbose)
	{
		cout << maze->toString() << endl;
		cout << task.getEnvironment()->moves() << endl;

		Tensor s = Tensor::Zero({ static_cast<int>(maze->getSensors().size()) });

		for (unsigned int i = 0; i < maze->mazeY(); i++)
		{
			for (unsigned int j = 0; j < maze->mazeX(); j++)
			{
				const int a = i * maze->mazeX() + j;
				Encoder::one_hot(s, a);

				p_network->activate(&s);
				//cout << s << endl;
				//cout << *p_network->get_output() << endl;

				switch (p_network->get_output()->max_value_index())
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

		for (unsigned int i = 0; i < maze->mazeY(); i++)
		{
			for (unsigned int j = 0; j < maze->mazeX(); j++)
			{
				const int a = i * maze->mazeX() + j;
				Encoder::one_hot(s, a);

				p_network->activate(&s);
				cout << *p_network->get_output() << endl;
			}
		}
	}

	int result = 0;

	if (task.isWinner())
	{
		result = 1;
	}

	return result;
}

void MazeExample::test_v(NeuralNetwork* p_network, bool p_verbose) const
{
	MazeTask task;
	Maze* maze = task.getEnvironment();
	task.getEnvironment()->reset();

	vector<float> sensors;
	Tensor action = Tensor::Zero({ 4 });
	Tensor state;

	Tensor s = Tensor::Zero({ static_cast<int>(maze->getSensors().size()) });

	for (unsigned int i = 0; i < maze->mazeY(); i++)
	{
		for (unsigned int j = 0; j < maze->mazeX(); j++)
		{
			const int a = i * maze->mazeX() + j;
			Encoder::one_hot(s, a);

			p_network->activate(&s);

			cout << p_network->get_output()[0] << " ";
		}

		cout << endl;
	}
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

void MazeExample::binary_encoding(const float p_value, Tensor* p_vector) {
	p_vector->fill(0);
	(*p_vector)[p_value] = 1.f;
}
