#include "FFN.h"
#include "NeuralNetwork.h"
#include "InputLayer.h"
#include "CoreLayer.h"

using namespace Coeus;

FFN::FFN()
{
	NeuralNetwork network;

	network.add_layer(new InputLayer("input", 2));
	network.add_layer(new CoreLayer("hidden", 4, NeuralGroup::ACTIVATION::SIGMOID));
	network.add_layer(new CoreLayer("output", 1, NeuralGroup::ACTIVATION::SIGMOID));

	network.add_connection("input", "hidden", Connection::INIT::UNIFORM, 0.1);
	network.add_connection("hidden", "output", Connection::INIT::UNIFORM, 0.1);

	cout << endl;
}


FFN::~FFN()
{
}
