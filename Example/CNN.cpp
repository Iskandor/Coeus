#include "CNN.h"
#include "ConvLayer.h"
#include "CoreLayer.h"

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
	Tensor input({ 2,5,5 }, Tensor::VALUE, 1);

	_network.add_layer(new ConvLayer("hidden0", RELU, new TensorInitializer(DEBUG), 3, 3, 1, 1, { 2,5,5 }));
	_network.add_layer(new CoreLayer("hidden1", 2, SOFTMAX, new TensorInitializer(DEBUG)));
	_network.add_connection("hidden0", "hidden1");
	_network.init();

	_network.activate(&input);

	cout << *_network.get_output() << endl;
}