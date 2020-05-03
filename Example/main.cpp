#include "FFN.h"
#include <cstdio>
#include "mountain_car_experiment.h"
#include "tensor_initializer.h"
#include <iostream>

#include "cnpy.h"
#include "maze_experiment.h"
#include "tensor_operator_cpu.h"

using namespace std;



int main()
{
	//FFN model;
	//model.run();

	/*
	tensor input({ 2, 4 }, tensor::VALUE, 1);
	input[2] = 2.f;
	activation_function* af = activation_function::softmax();
	cout << input << endl;
	input = af->forward(input);
	cout << input << endl;
	*/
	/*
	tensor m({ 10 }, tensor::VALUE, 0.5f);
	tensor n({ 100 }, tensor::VALUE, 0.2f);

	tensor_operator_cpu::add(m.data(), m.size(), n.data(), n.size(), n.data());
	cout << n << endl;
	*/
	
	mountain_car_experiment experiment;
	//experiment.simple_ddpg(2000);
	//experiment.simple_cacla(5000);
	//experiment.simple_dqn(2000);
	//experiment.run_cacla(10000);
	//for (int i = 0; i < 10; i++) experiment.run_ddpg(i, 500);
	//for (int i = 0; i < 10; i++) experiment.run_ddpg_fm(i, 500);
	for (int i = 2; i < 10; i++) experiment.run_ddpg_su(i, 500);

	//maze_experiment experiment;

	//experiment.run_qlearning(5000);
	//experiment.run_sarsa(5000);
	//experiment.run_dqn(1000);

	getchar();

	return 0;
}
