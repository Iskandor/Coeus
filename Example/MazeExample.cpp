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

	return test(&network, true);
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

	return test(&network, true);
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

			reward = task.getReward() + count_module.get_reward(&state1);
			td_error = critic.train(&state0, &state1, reward);

			actor.train(&state0, action0, td_error);

			state0.override(&state1);
		}

		cout << task.getEnvironment()->moves() << endl;

		if (reward > 0) {
			wins++;
		}
		else {
			loses++;
		}

		exploration.update(e);

		cout << wins << " / " << loses << endl;
	}

	test(&network_actor);
}

int MazeExample::example_deep_q(int p_hidden, float p_alpha, float p_lambda, const bool p_verbose) {
	MazeTask task;
	Maze* maze = task.getEnvironment();

	NeuralNetwork network;

	network.add_layer(new CoreLayer("hidden0", 256, RELU, new TensorInitializer(LECUN_UNIFORM), 16));
	network.add_layer(new CoreLayer("output", 4, LINEAR, new TensorInitializer(LECUN_UNIFORM)));
	// feed-forward connections
	network.add_connection("hidden0", "output");
	network.init();

	//BackProp optimizer(&network);
	//optimizer.init(new QuadraticCost(), p_alpha, 0.9, true);
	ADAM optimizer(&network);
	//RMSProp optimizer(&network);
	optimizer.init(new QuadraticCost(), p_alpha);
	//optimizer.add_learning_rate_module(new WarmStartup(1e-3, 1e-2, 10, 2));
	//CountModule curiosity(maze->mazeX() * maze->mazeY());
	//QLearning agent(&network, ADAM_RULE, p_alpha, 0.9, p_lambda);
	DeepQLearning agent(&network, &optimizer, 0.9f, 1024, 64);

	vector<float> sensors;
	Tensor state0, state1;
	float reward = 0;
	float epsilon = 0.3;
	int epochs = 0;

	int wins = 0, loses = 0;

	BoltzmanExploration exploration(1.f);

	//FILE* pFile = fopen("application.log", "w");
	//Output2FILE::Stream() = pFile;
	//FILELog::ReportingLevel() = FILELog::FromString("DEBUG1");

	//test(&network);

	//for (int e = 0; e < epochs; e++) {
	while (wins < 100 && loses < 10 && epochs < 100000) {
		epochs++;
		//if (p_verbose) cout << "Epoch " << e << endl;

		task.getEnvironment()->reset();

		sensors = maze->getSensors();
		state0 = encode_state(&sensors);

		while (!task.isFinished()) {
			//cout << maze->toString() << endl;
			network.activate(&state0);
			//const int action0 = choose_action(network.get_output(), epsilon);
			const int action0 = exploration.get_action(network.get_output());

			maze->performAction(action0);

			sensors = maze->getSensors();
			state1 = encode_state(&sensors);
			//curiosity.update(&state1);
			reward = task.getReward(); // +curiosity.get_reward(&state1);

		   /*
		   cout << maze->toString() << endl;
		   cout << state0 << endl;
		   cout << action0 << endl;
		   cout << state1 << endl;
		   cout << reward << endl;
		   */

			agent.train(&state0, action0, &state1, reward, task.isFinished());
			state0.override(&state1);
		}

		if (task.isWinner()) {
			wins++;
			if (epsilon > 0) epsilon *= 0.999f;
		}
		else {
			wins = 0;
			if (epsilon < 1e-1)	loses++;
		}

		//cout << epsilon << endl;
		//cout << wins << " " << task.getEnvironment()->moves() << " " << reward << endl;

		//agent.reset_traces();
		if (p_verbose)
		{
			cout << task.getEnvironment()->moves() << endl;
			cout << epsilon << endl;
			cout << wins << " / " << loses << endl;
		}

		//cout << maze->toString() << endl;
		//FILE_LOG(logDEBUG1) << wins << " " << loses;


		//exploration->update((float)e / epochs);
		/*
		if (epsilon > 0.1) {
		epsilon -= (1.0 / epochs);
		}
		*/
	}

	cout << epochs << endl;

	return test(&network, true);
}

void MazeExample::example_icm() {
	MazeTask task;
	Maze* maze = task.getEnvironment();

	NeuralNetwork network;

	network.add_layer(new CoreLayer("hidden0", 256, RELU, new TensorInitializer(LECUN_UNIFORM), 16));
	network.add_layer(new CoreLayer("output", 4, LINEAR, new TensorInitializer(LECUN_UNIFORM)));
	network.add_connection("hidden0", "output");
	network.init();

	//RMSProp optimizer(&network);
	//optimizer.init(new QuadraticCost(), 0.001);
	QLearning agent(&network, RMSPROP_RULE, 0.001f, 0.9f);


	NeuralNetwork network_fm;
	network_fm.add_layer(new CoreLayer("hidden0", 512, RELU, new TensorInitializer(LECUN_UNIFORM), 20));
	network_fm.add_layer(new CoreLayer("output", 16, SOFTMAX, new TensorInitializer(LECUN_UNIFORM)));
	network_fm.add_connection("hidden0", "output");
	network_fm.init();

	RMSProp optimizer_fm(&network_fm);
	optimizer_fm.init(new QuadraticCost(), 0.001f);

	ICM icm(&network_fm, &optimizer_fm);

	vector<float> sensors;
	Tensor action = Tensor::Zero({ 4 });
	Tensor state0, state1;
	float reward = 0;
	float epsilon = 0.1f;
	int epochs = 2000;

	int wins = 0, loses = 0;

	//FILE* pFile = fopen("application.log", "w");
	//Output2FILE::Stream() = pFile;
	//FILELog::ReportingLevel() = FILELog::FromString("DEBUG1");

	for (int e = 0; e < epochs; e++) {
		cout << "Epoch " << e << endl;

		task.getEnvironment()->reset();
		float int_error = 0;

		while (!task.isFinished()) {
			//cout << maze->toString() << endl;

			sensors = maze->getSensors();
			state0 = encode_state(&sensors);
			network.activate(&state0);
			//action = exploration->chooseAction(network.getOutput());
			const int action_index = choose_action(network.get_output(), epsilon);
			Encoder::one_hot(action, action_index);
			maze->performAction(action_index);

			sensors = maze->getSensors();
			state1 = encode_state(&sensors);
			reward = task.getReward();
			int_error += icm.train(&state0, &action, &state1);
			reward += icm.get_intrinsic_reward();
			agent.train(&state0, action_index, &state1, reward);
		}

		cout << task.getEnvironment()->moves() << endl;
		cout << epsilon << endl;
		cout << int_error / task.getEnvironment()->moves() << endl;

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

int MazeExample::test(NeuralNetwork* p_network, const bool p_verbose) const
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
	}

	int result = 0;

	if (task.isWinner())
	{
		result = 1;
	}

	return result;
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
