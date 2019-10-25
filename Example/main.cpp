#include "FFN.h"
#include "IrisTest.h"
#include "RNN.h"
#include "MazeExample.h"
#include "Encoder.h"
#include <bitset>
#include "mnist_reader.hpp"
#include "CNN.h"
#include "ConvLayer.h"
#include "PoolingLayer.h"
#include "TensorOperator.h"
#include "IOUtils.h"
#include "ContinuousTest.h"

using namespace std;

int main()
{	
	//FFN model;
	//model.run();
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
	
	MazeExample example;
	//example.example_q(15000);
	//example.example_sarsa(30000);
	//example.example_double_q(15000);
	//example.example_actor_critic(1000);
	example.example_nac(1000);

	//example.example_deep_q(64, 1e-3, 0, true);
	//example.example_icm(64);
	//example.example_selector(64);

	//ContinuousTest test;

	//test.run(32);
	//test.run_cart_pole(50000);

	/*
	IrisTest iris;

	iris.init();
	iris.run(2000);
	iris.test();
	*/

	system("pause");

	return 0;
}
