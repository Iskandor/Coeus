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
	Tensor input = Tensor::Value({2}, 1);

	_network.activate(&input);
}
