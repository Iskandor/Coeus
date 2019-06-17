#include "CNN.h"
#include "ConvLayer.h"
#include "CoreLayer.h"
#include "BackProph.h"
#include "QuadraticCost.h"
#include "ADAM.h"
#include "PoolingLayer.h"

using namespace std;

CNN::CNN()
{
	/*
	const string MNIST_DATA_LOCATION = "./data/";
	// MNIST_DATA_LOCATION set by MNIST cmake config
	std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

	// Load MNIST data
	_dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

	std::cout << "Nbr of training images = " << _dataset.training_images.size() << std::endl;
	std::cout << "Nbr of training labels = " << _dataset.training_labels.size() << std::endl;
	std::cout << "Nbr of test images = " << _dataset.test_images.size() << std::endl;
	std::cout << "Nbr of test labels = " << _dataset.test_labels.size() << std::endl;
	*/
}


CNN::~CNN()
{
}

void CNN::run()
{
	vector<Tensor*> input;

	input.push_back(new Tensor({ 2,6,6 }, Tensor::VALUE, 1));
	input.push_back(new Tensor({ 2,6,6 }, Tensor::VALUE, 2));

	vector<Tensor*> target;

	Tensor* t = new Tensor({ 2 }, Tensor::VALUE, 1);
	(*t)[0] = 0;
	target.push_back(t);

	t = new Tensor({ 2 }, Tensor::VALUE, 1);
	(*t)[1] = 0;
	target.push_back(t);

	_network.add_layer(new ConvLayer("conv0", RELU, new TensorInitializer(LECUN_UNIFORM), 3, 3, 1, 1, { 2,6,6 }));
	_network.add_layer(new PoolingLayer("pool0", 2, 2));
	_network.add_layer(new ConvLayer("conv1", RELU, new TensorInitializer(LECUN_UNIFORM), 3, 3, 1, 1));
	_network.add_layer(new CoreLayer("output", 2, SOFTMAX, new TensorInitializer(LECUN_UNIFORM)));
	_network.add_connection("conv0", "pool0");
	_network.add_connection("pool0", "conv1");
	//_network.add_connection("conv0", "conv1");
	_network.add_connection("conv1", "output");
	_network.init();

	BackProp optimizer(&_network);
	optimizer.init(new QuadraticCost(), 0.1);


	for(int i = 0; i < 100; i++)
	{
		float error = 0;
		for(int s  = 0; s < 2; s++)
		{
			error += optimizer.train(input[s], target[s]);
		}

		cout << error << endl;
	}

	for (int s = 0; s < 2; s++)
	{
		_network.activate(input[s]);
		cout << *_network.get_output() << endl;
	}
}