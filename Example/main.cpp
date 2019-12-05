#include "FFN.h"
#include "CNN.h"
#include "RNN.h"
#include "ContinuousTest.h"
#include "MazeExample.h"

using namespace std;

int main()
{
	//FFN model;
	//model.run();
	//model.run_ubal();
	//model.run_iris();

	//RNN model;
	//model.run_pack();
	//model.run_pack2();
	//model.test_pack();
	//model.test_pack_cm();
	//model.test_pack_alt();
	//model.run_add_problem();
	//model.run_sin_prediction();
	//model.run_add_problem_gru();

	//CNN model;
	//model.run();
	//model.run_mnist();
	//model.test();
	
	//MazeExample example;
	//example.example_q(15000);
	//example.example_sarsa(30000);
	//example.example_double_q(15000);
	//example.example_deep_q(2000);
	//example.example_actor_critic(1000);
	//example.example_nac(1000);
	//example.example_a2c(1000);
	//example.example_a3c(1);
	//
	//example.example_icm(64);
	//example.example_selector(64);

	ContinuousTest test;

	//test.run_simple_cacla(1000);
	test.run_simple_ddpg(500);
	//test.run_cacla(50000);
	//test.run_ddpg(50000);

	/*
	IrisTest iris;

	iris.init();
	iris.run(2000);
	iris.test();
	*/

	system("pause");

	return 0;
}
