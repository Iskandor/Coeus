#include "FFN.h"
#include <cstdio>
#include "mountain_car_experiment.h"
#include "tensor_initializer.h"
#include <iostream>
#include "maze_experiment.h"
#include "tensor_operator_cpu.h"
using namespace std;

int main()
{
	//FFN model;
	//model.run();

	tensor input({ 2, 4 }, tensor::VALUE, 1);
	input[2] = 2.f;
	activation_function* af = activation_function::softmax();
	cout << input << endl;
	input = af->forward(input);
	cout << input << endl;

	mountain_car_experiment experiment;
	//experiment.simple_ddpg(2000);
	//experiment.simple_cacla(5000);
	//experiment.simple_dqn(2000);
	//experiment.run_ddpg(500);
	//experiment.run_ddpg_fm(500);

	//maze_experiment experiment;

	//experiment.run_qlearning(5000);
	//experiment.run_sarsa(5000);
	//experiment.run_dqn(1000);

	getchar();

	return 0;
}
