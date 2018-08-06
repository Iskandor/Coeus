#include "MazeExample.h"
#include "RandomGenerator.h"
#include <iostream>
#include "QLearning.h"
#include "MazeTask.h"
#include "ADAM.h"
#include "QuadraticCost.h"
#include "InputLayer.h"
#include "CoreLayer.h"
#include "RMSProp.h"
#include "SARSA.h"
#include "TD.h"
#include "Actor.h"
#include "BackProph.h"
#include "DoubleQLearning.h"
#include "DeepQLearning.h"

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

	network.add_layer(new InputLayer("input", 64));
	network.add_layer(new CoreLayer("hidden0", 512, NeuralGroup::RELU));
	network.add_layer(new CoreLayer("output", 4, NeuralGroup::LINEAR));
	// feed-forward connections
	network.add_connection("input", "hidden0", Connection::LECUN_UNIFORM);
	network.add_connection("hidden0", "output", Connection::LECUN_UNIFORM);
	network.init();

	//BackProp optimizer(&network);
	//optimizer.init(new QuadraticCost(), 0.01, 0.9, true);
	ADAM optimizer(&network);
	optimizer.init(new QuadraticCost(), 0.01);
	QLearning agent(&network, &optimizer, 0.9);

	vector<double> sensors;
	Tensor state0, state1;
	int action;
	double reward = 0;
	double epsilon = 1;
	int epochs = 4000;

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
			agent.train(&state0, action, &state1, reward);
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

void MazeExample::example_double_q() {
	MazeTask task;
	Maze* maze = task.getEnvironment();

	NeuralNetwork network_a;

	network_a.add_layer(new InputLayer("input", 64));
	network_a.add_layer(new CoreLayer("hidden0", 164, NeuralGroup::RELU));
	network_a.add_layer(new CoreLayer("output", 4, NeuralGroup::LINEAR));
	// feed-forward connections
	network_a.add_connection("input", "hidden0", Connection::LECUN_UNIFORM);
	network_a.add_connection("hidden0", "output", Connection::LECUN_UNIFORM);
	network_a.init();

	ADAM optimizer1(&network_a);
	optimizer1.init(new QuadraticCost(), 0.003);

	NeuralNetwork network_b;

	network_b.add_layer(new InputLayer("input", 64));
	network_b.add_layer(new CoreLayer("hidden0", 164, NeuralGroup::RELU));
	network_b.add_layer(new CoreLayer("output", 4, NeuralGroup::LINEAR));
	// feed-forward connections
	network_b.add_connection("input", "hidden0", Connection::LECUN_UNIFORM);
	network_b.add_connection("hidden0", "output", Connection::LECUN_UNIFORM);
	network_b.init();

	ADAM optimizer2(&network_b);
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

	network.add_layer(new InputLayer("input", 64));
	network.add_layer(new CoreLayer("hidden0", 512, NeuralGroup::RELU));
	network.add_layer(new CoreLayer("output", 4, NeuralGroup::LINEAR));
	// feed-forward connections
	network.add_connection("input", "hidden0", Connection::LECUN_UNIFORM);
	network.add_connection("hidden0", "output", Connection::LECUN_UNIFORM);
	network.init();

	ADAM optimizer(&network);
	optimizer.init(new QuadraticCost(), 0.01);
	SARSA agent(&network, &optimizer, 0.9);

	vector<double> sensors;
	Tensor state0, state1;
	int action0, action1;
	double reward = 0;
	double epsilon = 1;
	int epochs = 4000;

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

	network_critic.add_layer(new InputLayer("input", 64));
	network_critic.add_layer(new CoreLayer("hidden0", 256, NeuralGroup::RELU));
	network_critic.add_layer(new CoreLayer("output", 1, NeuralGroup::LINEAR));
	// feed-forward connections
	network_critic.add_connection("input", "hidden0", Connection::LECUN_UNIFORM);
	network_critic.add_connection("hidden0", "output", Connection::LECUN_UNIFORM);
	network_critic.init();

	ADAM optimizer1(&network_critic);
	optimizer1.init(new QuadraticCost(), 0.01);
	TD critic(&network_critic, &optimizer1, 0.9);

	NeuralNetwork network_actor;

	network_actor.add_layer(new InputLayer("input", 64));
	network_actor.add_layer(new CoreLayer("hidden0", 512, NeuralGroup::RELU));
	network_actor.add_layer(new CoreLayer("output", 4, NeuralGroup::LINEAR));
	// feed-forward connections
	network_actor.add_connection("input", "hidden0", Connection::LECUN_UNIFORM);
	network_actor.add_connection("hidden0", "output", Connection::LECUN_UNIFORM);
	network_actor.init();

	ADAM optimizer2(&network_actor);
	optimizer2.init(new QuadraticCost(), 0.01);
	Actor actor(&network_actor, &optimizer2, 0.9);

	vector<double> sensors;
	Tensor state0, state1;
	int action0;
	int value0, value1;
	double reward = 0;
	double epsilon = 1;
	int epochs = 4000;

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

	network.add_layer(new InputLayer("input", 64));
	network.add_layer(new CoreLayer("hidden0", 256, NeuralGroup::RELU));
	network.add_layer(new CoreLayer("output", 4, NeuralGroup::LINEAR));
	// feed-forward connections
	network.add_connection("input", "hidden0", Connection::LECUN_UNIFORM);
	network.add_connection("hidden0", "output", Connection::LECUN_UNIFORM);
	network.init();

	//BackProp optimizer(&network);
	//optimizer.init(new QuadraticCost(), 0.01, 0.9, true);
	ADAM optimizer(&network);
	optimizer.init(new QuadraticCost(), 0.01);
	DeepQLearning agent(&network, &optimizer, 0.9, 256, 32);

	vector<double> sensors;
	Tensor state0, state1;
	int action;
	double reward = 0;
	double epsilon = 1;
	int epochs = 500;

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
}

Tensor MazeExample::encode_state(vector<double>* p_sensors) {
	const Tensor res({ 64 }, Tensor::ZERO);
	Tensor encoded({ 4 }, Tensor::ZERO);

	for (unsigned int i = 0; i < p_sensors->size(); i++) {
		if (p_sensors->at(i) > 0) {
			binary_encoding(p_sensors->at(i) - 1, &encoded);
		}
		else {
			encoded.fill(0);
		}

		for (int j = 0; j < 4; j++) {
			res[i * 4 + j] = encoded[j];
		}
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
