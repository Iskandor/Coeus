#include "MazeExample.h"
#include "RandomGenerator.h"
#include <iostream>
#include "QLearning.h"
#include "MazeTask.h"
#include "QuadraticCost.h"
#include "InputLayer.h"
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
#include "BackProph.h"
#include "QLearning2.h"

using namespace Coeus;

MazeExample::MazeExample()
{
}


MazeExample::~MazeExample()
{
}

void MazeExample::example_q() {
	MazeTask task;
	Maze* maze = task.getEnvironment();

	NeuralNetwork network;

	network.add_layer(new InputLayer("input", 16));
	network.add_layer(new CoreLayer("hidden", 256, RELU));
	network.add_layer(new CoreLayer("output", 4, LINEAR));
	// feed-forward connections
	network.add_connection("input", "hidden", Connection::UNIFORM, 1e-2);
	network.add_connection("hidden", "output", Connection::UNIFORM, 1e-2);
	network.init();

	//BackProp optimizer(&network);
	//optimizer.init(new QuadraticCost(), 0.1, 0.9, false);
	//ADAM optimizer(&network);
	//RMSProp optimizer(&network);
	//optimizer.init(new QuadraticCost(), 1e-3);
	//optimizer.add_learning_rate_module(new WarmStartup(1e-4, 1e-3, 10, 2));
	//QLearning agent(&network, &optimizer, 0.9);
	QLearning2 agent(&network, 1e-3, 0.9);

	vector<double> sensors;
	Tensor state0, state1;
	double reward = 0;
	double epsilon = 1;
	const int epochs = 3000;

	int wins = 0, loses = 0;

	//FILE* pFile = fopen("application.log", "w");
	//Output2FILE::Stream() = pFile;
	//FILELog::ReportingLevel() = FILELog::FromString("DEBUG1");

	//test(&network);

	for (int e = 0; e < epochs; e++) {
		//cout << "Epoch " << e << endl;

		task.getEnvironment()->reset();

		sensors = maze->getSensors();
		state0 = encode_state(&sensors);

		while (!task.isFinished()) {
			//cout << maze->toString() << endl;
			network.activate(&state0);
			const int action0 = choose_action(network.get_output(), epsilon);
			maze->performAction(action0);

			sensors = maze->getSensors();
			state1 = encode_state(&sensors);
			reward = task.getReward();

			/*
			cout << task.getEnvironment()->toString() << endl;
			cout << reward << endl;
			*/

			agent.train(&state0, action0, &state1, reward);
			state0.override(&state1);
		}

		//agent.reset_traces();
		//cout << task.getEnvironment()->moves() << endl;
		//cout << epsilon << endl;

		if (task.isWinner()) {
			wins++;
		}
		else {
			loses++;
		}

		//cout << maze->toString() << endl;
		//cout << wins << " / " << loses << endl;
		//FILE_LOG(logDEBUG1) << wins << " " << loses;


		//exploration->update((double)e / epochs);

		if (epsilon > 0.1) {
			epsilon -= (1.0 / epochs);
		}
	}

	test(&network);
}

void MazeExample::example_double_q() {
	MazeTask task;
	Maze* maze = task.getEnvironment();

	NeuralNetwork network_a;

	network_a.add_layer(new InputLayer("input", 64));
	network_a.add_layer(new CoreLayer("hidden0", 164, RELU));
	network_a.add_layer(new CoreLayer("output", 4, LINEAR));
	// feed-forward connections
	network_a.add_connection("input", "hidden0", Connection::LECUN_UNIFORM);
	network_a.add_connection("hidden0", "output", Connection::LECUN_UNIFORM);
	network_a.init();

	RMSProp optimizer1(&network_a);
	optimizer1.init(new QuadraticCost(), 0.003);

	NeuralNetwork network_b;

	network_b.add_layer(new InputLayer("input", 64));
	network_b.add_layer(new CoreLayer("hidden0", 164, RELU));
	network_b.add_layer(new CoreLayer("output", 4, LINEAR));
	// feed-forward connections
	network_b.add_connection("input", "hidden0", Connection::LECUN_UNIFORM);
	network_b.add_connection("hidden0", "output", Connection::LECUN_UNIFORM);
	network_b.init();

	RMSProp optimizer2(&network_b);
	optimizer2.init(new QuadraticCost(), 0.003);

	DoubleQLearning critic(&network_a, &optimizer1, &network_b, &optimizer2, 0.9);

	vector<double> sensors;
	Tensor state0, state1;
	Tensor output;
	int action0;
	double reward = 0;
	double epsilon = 1;
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

			output = (*network_a.get_output() + *network_b.get_output()) / 2;
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


		//exploration->update((double)e / epochs);

		if (epsilon > 0.1) {
			epsilon -= (1.0 / epochs);
		}
	}
}

void MazeExample::example_sarsa() {
	MazeTask task;
	Maze* maze = task.getEnvironment();

	NeuralNetwork network;

	network.add_layer(new InputLayer("input", 16));
	network.add_layer(new CoreLayer("hidden", 64, RELU));
	network.add_layer(new CoreLayer("hidden1", 16, RELU));
	network.add_layer(new CoreLayer("output", 4, LINEAR));
	// feed-forward connections
	network.add_connection("input", "hidden", Connection::LECUN_UNIFORM);
	network.add_connection("hidden", "hidden1", Connection::LECUN_UNIFORM);
	network.add_connection("hidden1", "output", Connection::LECUN_UNIFORM);
	network.init();

	RMSProp optimizer(&network);
	optimizer.init(new QuadraticCost(), 0.001);
	SARSA agent(&network, &optimizer, 0.9);

	vector<double> sensors;
	Tensor state0, state1;
	int action0, action1;
	double reward = 0;
	double epsilon = 1;
	int epochs = 6000;

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
			network.activate(&state0);
			//action = exploration->chooseAction(network.getOutput());
			action0 = choose_action(network.get_output(), epsilon);
			maze->performAction(action0);

			sensors = maze->getSensors();
			state1 = encode_state(&sensors);
			reward = task.getReward();
			network.activate(&state1);
			action1 = choose_action(network.get_output(), epsilon);
			agent.train(&state0, action0, &state1, action1, reward);
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


		//exploration->update((double)e / epochs);

		if (epsilon > 0.1) {
			epsilon -= (1.0 / epochs);
		}
	}
}

void MazeExample::example_actor_critic() {
	MazeTask task;
	Maze* maze = task.getEnvironment();

	NeuralNetwork network_critic;

	network_critic.add_layer(new InputLayer("input", 16));
	network_critic.add_layer(new CoreLayer("hidden0", 128, RELU));
	network_critic.add_layer(new CoreLayer("output", 1, LINEAR));
	// feed-forward connections
	network_critic.add_connection("input", "hidden0", Connection::LECUN_UNIFORM);
	network_critic.add_connection("hidden0", "output", Connection::LECUN_UNIFORM);
	network_critic.init();

	RMSProp optimizer1(&network_critic);
	optimizer1.init(new QuadraticCost(), 0.001);
	TD critic(&network_critic, &optimizer1, 0.9);

	NeuralNetwork network_actor;

	network_actor.add_layer(new InputLayer("input", 16));
	network_actor.add_layer(new CoreLayer("hidden0", 256, RELU));
	network_actor.add_layer(new CoreLayer("output", 4, SOFTMAX));
	// feed-forward connections
	network_actor.add_connection("input", "hidden0", Connection::LECUN_UNIFORM);
	network_actor.add_connection("hidden0", "output", Connection::LECUN_UNIFORM);
	network_actor.init();

	RMSProp optimizer2(&network_actor);
	optimizer2.init(new QuadraticCost(), 0.001);
	Actor actor(&network_actor, &optimizer2, 0.9);

	vector<double> sensors;
	Tensor state0, state1;
	int action0;
	int value0, value1;
	double reward = 0;
	double epsilon = 1;
	int epochs = 10000;

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
			network_critic.activate(&state0);
			network_actor.activate(&state0);
			value0 = network_critic.get_output()->at(0);
			action0 = choose_action(network_actor.get_output(), epsilon);
			maze->performAction(action0);

			sensors = maze->getSensors();
			state1 = encode_state(&sensors);
			network_critic.activate(&state1);
			value1 = network_critic.get_output()->at(0);
			reward = task.getReward();
			critic.train(&state0, &state1, reward);
			actor.train(&state0, action0, value0, value1, reward);
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


		//exploration->update((double)e / epochs);

		if (epsilon > 0.1) {
			epsilon -= (1.0 / epochs);
		}
	}
}

void MazeExample::example_deep_q() {
	MazeTask task;
	Maze* maze = task.getEnvironment();

	NeuralNetwork network;

	network.add_layer(new InputLayer("input", 16));
	network.add_layer(new CoreLayer("hidden0", 256, RELU));
	network.add_layer(new CoreLayer("output", 4, LINEAR));
	// feed-forward connections
	network.add_connection("input", "hidden0", Connection::LECUN_UNIFORM);
	network.add_connection("hidden0", "output", Connection::LECUN_UNIFORM);
	network.init();

	//BackProp optimizer(&network);
	//optimizer.init(new QuadraticCost(), 0.01, 0.9, true);
	ADAM optimizer(&network);
	optimizer.init(new QuadraticCost(), 1e-3);
	DeepQLearning agent(&network, &optimizer, 0.9, 64, 8);

	vector<double> sensors;
	Tensor state0, state1;
	int action;
	double reward = 0;
	double epsilon = 1;
	int epochs = 2000;

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
			network.activate(&state0);
			//action = exploration->chooseAction(network.getOutput());
			action = choose_action(network.get_output(), epsilon);
			maze->performAction(action);

			sensors = maze->getSensors();
			state1 = encode_state(&sensors);
			reward = task.getReward();
			agent.train(&state0, action, &state1, reward, task.isFinished());
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


		//exploration->update((double)e / epochs);

		if (epsilon > 0.1) {
			epsilon -= (1.0 / epochs);
		}
	}

	test(&network);
}

void MazeExample::example_icm() {
	MazeTask task;
	Maze* maze = task.getEnvironment();

	NeuralNetwork network;

	network.add_layer(new InputLayer("input", 16));
	network.add_layer(new CoreLayer("hidden0", 256, RELU));
	network.add_layer(new CoreLayer("output", 4, LINEAR));
	network.add_connection("input", "hidden0", Connection::LECUN_UNIFORM);
	network.add_connection("hidden0", "output", Connection::LECUN_UNIFORM);
	network.init();

	RMSProp optimizer(&network);
	optimizer.init(new QuadraticCost(), 0.001);
	QLearning agent(&network, &optimizer, 0.9);


	NeuralNetwork network_fm;
	network_fm.add_layer(new InputLayer("input", 20));
	network_fm.add_layer(new CoreLayer("hidden0", 512, RELU));
	network_fm.add_layer(new CoreLayer("output", 16, SOFTMAX));
	network_fm.add_connection("input", "hidden0", Connection::LECUN_UNIFORM);
	network_fm.add_connection("hidden0", "output", Connection::LECUN_UNIFORM);
	network_fm.init();

	RMSProp optimizer_fm(&network_fm);
	optimizer_fm.init(new QuadraticCost(), 0.001);

	ICM icm(&network_fm, &optimizer_fm);

	vector<double> sensors;
	Tensor action = Tensor::Zero({ 4 });
	Tensor state0, state1;
	double reward = 0;
	double epsilon = 0.1;
	int epochs = 2000;

	int wins = 0, loses = 0;

	//FILE* pFile = fopen("application.log", "w");
	//Output2FILE::Stream() = pFile;
	//FILELog::ReportingLevel() = FILELog::FromString("DEBUG1");

	for (int e = 0; e < epochs; e++) {
		cout << "Epoch " << e << endl;

		task.getEnvironment()->reset();
		double int_error = 0;

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


		//exploration->update((double)e / epochs);

		if (epsilon > 0.1) {
			epsilon -= (1.0 / epochs);
		}
	}
}

void MazeExample::test(NeuralNetwork* p_network) const
{
	MazeTask task;
	Maze* maze = task.getEnvironment();
	task.getEnvironment()->reset();

	vector<double> sensors;
	Tensor action = Tensor::Zero({ 4 });
	Tensor state;

	int step = 0;

	cout << "--- TEST ---" << endl;

	while (!task.isFinished() && step < 20) {
		sensors = maze->getSensors();
		state = encode_state(&sensors);
		p_network->activate(&state);
		const int action_index = choose_action(p_network->get_output(), 0);
		Encoder::one_hot(action, action_index);
		maze->performAction(action_index);
		step++;		
	}

	cout << maze->toString() << endl;
	cout << task.getEnvironment()->moves() << endl;
	cout << task.getReward() << endl;

	Tensor s = Tensor::Zero({ static_cast<int>(maze->getSensors().size()) });

	for(int i = 0; i < maze->mazeY(); i++)
	{
		for (int j = 0; j < maze->mazeX(); j++)
		{
			const int a = i * maze->mazeX() + j;
			Encoder::one_hot(s, a);

			p_network->activate(&s);
			//cout << s << endl;
			//cout << *p_network->get_output() << endl;

			switch(p_network->get_output()->max_value_index())
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
				default: ;
			}
		}

		cout << endl;
	}
}

Tensor MazeExample::encode_state(vector<double>* p_sensors) {
	const Tensor res({ static_cast<int>(p_sensors->size()) }, Tensor::ZERO);

	for (unsigned int i = 0; i < p_sensors->size(); i++) {
		res[i] = p_sensors->at(i);
	}

	return Tensor(res);
}

int MazeExample::choose_action(Tensor* p_input, const double epsilon) {
	int action = 0;
	const double random = RandomGenerator::getInstance().random();

	if (random < epsilon) {
		action = RandomGenerator::getInstance().random(0, 3);
	}
	else {
		for (int i = 1; i < p_input->size(); i++) {
			if ((*p_input)[i] != (*p_input)[i])
			{
				assert(0);
			}
			if ((*p_input)[i] >(*p_input)[action]) {
				action = i;
			}
		}
	}

	return action;
}

void MazeExample::binary_encoding(const double p_value, Tensor* p_vector) {
	p_vector->fill(0);
	(*p_vector)[p_value] = 1;
}
