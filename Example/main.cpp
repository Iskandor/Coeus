
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

	/*
	Tensor m({ 3,5,4 }, Tensor::VALUE, 2);
	Tensor n({ 3,4 }, Tensor::RANDOM, 1);

	cout << m << endl;

	//n.replicate(3);
	TensorOperator::instance().V_reduce(n.arr(), m.arr(), 3, 5, 4, 1);

	cout << n << endl;
	*/

	//CNN model;
	//model.run();
	//model.run_mnist();
	//model.test();
	
	MazeExample example;
	//example.example_q(64, 1e-3, 0, true);
	//example.example_sarsa(64, 1e-3, 0, true);
	example.example_double_q(64, 1e-3);
	//example.example_actor_critic(64);
	//example.example_deep_q(64, 1e-3, 0, true);
	//example.example_icm(64);
	//example.example_selector(64);

	/*
	IrisTest iris;

	iris.init();
	iris.run(2000);
	iris.test();
	*/

	system("pause");

	return 0;
}
