#include "RNN.h"
#include "InputLayer.h"
#include "LSTMLayer.h"
#include "CoreLayer.h"
#include "QuadraticCost.h"
#include "ADAM.h"
#include "BPTT.h"


RNN::RNN()
{
	_network.add_layer(new InputLayer("input", 1));
	_network.add_layer(new LSTMLayer("hidden", 4, SIGMOID));
	_network.add_layer(new CoreLayer("output", 1, SIGMOID));

	_network.add_connection("input", "hidden", Connection::UNIFORM, 0.1);
	_network.add_connection("hidden", "output", Connection::UNIFORM, 0.1);

	_network.init();
}


RNN::~RNN()
{
}

void RNN::run()
{
	double data_i[8]{ 0,0,0,1,1,0,1,1 };
	double data_t[4]{ 0,1,1,0 };

	Tensor input[8];
	Tensor target[4];

	for (int i = 0; i < 8; i++) {
		double *d = Tensor::alloc_arr(1);
		d[0] = data_i[i];
		input[i] = Tensor({ 1 }, d);
	}

	for (int i = 0; i < 4; i++) {
		double *t = Tensor::alloc_arr(1);
		t[0] = data_t[i];
		target[i] = Tensor({ 1 }, t);
	}

	//BackProp model(&_network);
	//RMSProp model(&_network);
	//AdaMax model(&_network);
	ADAM algorithm(&_network);
	//AMSGrad model(&_network);
	//Nadam model(&_network);

	//model.init(new QuadraticCost(), 0.1, 0.99, true);
	algorithm.init(new QuadraticCost(), 0.05);
	/*
	for (int t = 0; t < 4000; t++) {
		double error = 0;
		for (int i = 0; i < 4; i++) {
			error += algorithm.train(&input[i], &target[i]);
		}
		cout << error << endl;
	}
	*/

	cout << endl;

	for (int i = 0; i < 8; i++) {
		_network.activate(&input[i]);
		cout << _network.get_output()->at(0) << endl;
	}
}
