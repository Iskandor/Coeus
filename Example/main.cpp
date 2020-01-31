#include "FFN.h"
#include "CNN.h"
#include "RNN.h"
#include "ContinuousTest.h"
#include "MazeExample.h"
#include "MotivationTest.h"

using namespace std;

int main()
{
	/*
	Tensor m({ 2,2 }, Tensor::RANDOM);
	m[0] = 1;
	m[1] = 1;
	m[2] = 0;
	m[3] = 0;
	
	cout << m << endl;
	
	Tensor n = m.pinv();

	cout << n << endl;
	*/
	
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
	//example.example_deep_q(1000);
	//example.example_actor_critic(1000);
	//example.example_nac(1000);
	//example.example_a2c(1000);
	//example.example_a3c(2000);
	//
	//example.example_icm(64);
	//example.example_selector(64);

	ContinuousTest test;

	//test.run_simple_cacla(1000);
	//test.run_simple_cacer(1000);
	//test.run_simple_ddpg(1000);
	//test.run_cacla(10000);
	for(int i = 0; i < 10; i++) test.run_ddpg(100, true);

	/*
	IrisTest iris;

	iris.init();
	iris.run(2000);
	iris.test();
	*/

	//MotivationTest test;

	//test.cart_pole_icm(10000);
	//test.test_icm(1000);
	//test.test_gm2(1000);

	system("pause");

	return 0;
}
