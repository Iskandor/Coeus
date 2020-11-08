#include "FFN.h"
#include <cstdio>
#include "mountain_car_experiment.h"
#include "tensor_initializer.h"
#include <iostream>


#include "cart_pole_experiment.h"
#include "cnpy.h"
#include "maze_experiment.h"
#include "tensor_operator_cpu.h"

using namespace std;



int main()
{
	//FFN model;
	//model.run();
	
	mountain_car_experiment experiment;
	//for (int i = 0; i < 10; i++) experiment.run_ddpg(i, 1000);
	//for (int i = 0; i < 10; i++) experiment.run_ddpg_fm(i, 1000);
	for (int i = 0; i < 10; i++) experiment.run_ddpg_su(i, 1000);

	//cart_pole_experiment experiment;
	//for (int i = 0; i < 3; i++) experiment.run_ddpg(i, 5000);
	//for (int i = 0; i < 3; i++) experiment.run_ddpg_fm(i, 5000);
	//for (int i = 0; i < 3; i++) experiment.run_ddpg_su(i, 5000);

	//maze_experiment experiment;
	//experiment.run_qlearning(5000);
	//experiment.run_sarsa(5000);
	//experiment.run_dqn(1000);
	//experiment.run_ac(1000);
	//experiment.run_ppo(1000);

	getchar();

	return 0;
}
