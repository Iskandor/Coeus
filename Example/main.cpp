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

	//mountain_car_experiment experiment;
	//experiment.test_simple(1000);

	maze_experiment experiment;

	experiment.run_qlearning(5000);
	//experiment.run_sarsa(5000);
	//experiment.run_dqn(500);

	getchar();

	return 0;
}
