#include "FFN.h"
#include "NeuralNetwork.h"
#include "InputLayer.h"
#include "CoreLayer.h"

FFN::FFN()
{
	_network.add_layer(new InputLayer("input", 2));
	_network.add_layer(new CoreLayer("hidden", 4, NeuralGroup::ACTIVATION::SIGMOID));
	_network.add_layer(new CoreLayer("output", 1, NeuralGroup::ACTIVATION::SIGMOID, false));

	_network.add_connection("input", "hidden", Connection::INIT::UNIFORM, 0.1);
	_network.add_connection("hidden", "output", Connection::INIT::UNIFORM, 0.1);
}


FFN::~FFN()
{
}

void FFN::run() {
	double data_i[8]{ 0,0,0,1,1,0,1,1 };
	double data_t[4]{ 0,0,0,1 };

	Tensor* input[4];
	Tensor* target[4];

	for (int i = 0; i < 4; i++) {
		double* d = new double[2];

		d[0] = data_i[i * 2];
		d[1] = data_i[i * 2 + 1];

		double* t = new double[1];
		t[0] = data_t[i];

		input[i] = new Tensor({ 2 }, d);
		target[i] = new Tensor({ 1 }, t);
	}

	/*
	for (int i = 0; i < 4; i++) {
		_network.activate(input[i]);
	}
	*/

	for (int i = 0; i < 4; i++) {
		delete input[i];
		delete target[i];
	}
}
