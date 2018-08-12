#include "RNN.h"
#include "InputLayer.h"
#include "LSTMLayer.h"
#include "CoreLayer.h"
#include "QuadraticCost.h"
#include "ADAM.h"
#include "BPTT.h"


RNN::RNN()
{
	_network.add_layer(new InputLayer("input0", 1));
	_network.add_layer(new InputLayer("input1", 1));
	_network.add_layer(new LSTMLayer("hidden0", 2, NeuralGroup::ACTIVATION::SIGMOID));
	_network.add_layer(new LSTMLayer("hidden1", 2, NeuralGroup::ACTIVATION::SIGMOID));
	_network.add_layer(new CoreLayer("output", 1, NeuralGroup::ACTIVATION::SIGMOID));

	_network.add_connection("input0", "hidden0", Connection::UNIFORM, 0.1);
	_network.add_connection("input1", "hidden1", Connection::UNIFORM, 0.1);
	_network.add_connection("hidden0", "hidden1", Connection::UNIFORM, 0.1);
	_network.add_connection("hidden1", "output", Connection::UNIFORM, 0.1);

	_network.init();
}


RNN::~RNN()
{
}

void RNN::run()
{
	double data_i[8]{ 0,0,0,1,1,0,1,1 };
	double data_t[4]{ 0,1,1,0 };

	vector<Tensor*> input[4];
	Tensor target[4];

	for (int i = 0; i < 4; i++) {
		double *d0 = Tensor::alloc_arr(1);
		d0[0] = data_i[i * 2];
		double *d1 = Tensor::alloc_arr(1);
		d1[0] = data_i[i * 2 + 1];

		double *t = Tensor::alloc_arr(1);
		t[0] = data_t[i];

		input[i].push_back(new Tensor({ 1 }, d0));
		input[i].push_back(new Tensor({ 1 }, d1));
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

	BPTT model(&_network, &algorithm);

	for (int t = 0; t < 4000; t++) {
		double error = 0;
		for (int i = 0; i < 4; i++) {
			error += model.train(&input[i], &target[i]);
		}
		cout << error << endl;
	}

	cout << endl;

	for (int i = 0; i < 4; i++) {
		_network.activate(&input[i]);
		cout << _network.get_output()->at(0) << endl;
	}
}
