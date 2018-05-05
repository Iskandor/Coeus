#include "FFN.h"
#include "NeuralNetwork.h"
#include "InputLayer.h"
#include "CoreLayer.h"
#include "BackProp.h"
#include "QuadraticCost.h"

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

	Tensor input[4];
	Tensor target[4];

	for (int i = 0; i < 4; i++) {
		double *d = Tensor::alloc_arr(2);

		d[0] = data_i[i * 2];
		d[1] = data_i[i * 2 + 1];

		double *t = Tensor::alloc_arr(1);
		t[0] = data_t[i];

		input[i] = Tensor({ 2 }, d);
		target[i] = Tensor({ 1 }, t);
	}

	BackProp model(&_network);

	model.init(new QuadraticCost(), 0.05);

	for(int t = 0; t < 10000; t++) {
		double error = 0;
		for (int i = 0; i < 4; i++) {
			error += model.train(&input[i], &target[i]);
		}
		cout << error << endl;
	}

	for (int i = 0; i < 4; i++) {
		_network.activate(&input[i]);
		cout << _network.get_output()->at(i) << endl;
	}
}
