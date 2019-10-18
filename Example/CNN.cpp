#include "CNN.h"
#include "ConvLayer.h"
#include "CoreLayer.h"
#include "BackProph.h"
#include "QuadraticCost.h"
#include "ADAM.h"
#include "PoolingLayer.h"
#include "RMSProp.h"
#include "TensorOperator.h"
#include <chrono>
#include "CrossEntropyCost.h"

using namespace std;

CNN::CNN()
{
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

void CNN::run_mnist()
{
	const string MNIST_DATA_LOCATION = "./data/";
	// MNIST_DATA_LOCATION set by MNIST cmake config
	std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

	// Load MNIST data
	_dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

	std::cout << "Nbr of training images = " << _dataset.training_images.size() << std::endl;
	std::cout << "Nbr of training labels = " << _dataset.training_labels.size() << std::endl;
	std::cout << "Nbr of test images = " << _dataset.test_images.size() << std::endl;
	std::cout << "Nbr of test labels = " << _dataset.test_labels.size() << std::endl;
	
	_train_input.reserve(_dataset.training_images.size());
	for (const auto& training_image : _dataset.training_images)
	{
		Tensor* t = new Tensor({ 1, 28, 28 }, training_image);

		TensorOperator::instance().vc_prod(t->arr(), 1.f/256, t->arr(), t->size());
		_train_input.push_back(t);
	}

	_train_target.reserve(_dataset.training_images.size());
	for (int label : _dataset.training_labels)
	{
		Tensor* t = new Tensor({ 10 }, Tensor::ZERO);
		t->set(label, 1);
		_train_target.push_back(t);

	}

	_network.add_layer(new ConvLayer("conv0", RELU, new TensorInitializer(LECUN_UNIFORM), 8, 5, 1, 2, { 1,28,28 }));
	_network.add_layer(new PoolingLayer("pool0", 2, 2));
	_network.add_layer(new ConvLayer("conv1", RELU, new TensorInitializer(LECUN_UNIFORM), 16, 5, 1, 2));
	_network.add_layer(new PoolingLayer("pool1", 3, 3));
	_network.add_layer(new CoreLayer("output", 10, SOFTMAX, new TensorInitializer(LECUN_UNIFORM)));

	_network.add_connection("conv0", "pool0");
	_network.add_connection("pool0", "conv1");
	_network.add_connection("conv1", "pool1");
	_network.add_connection("pool1", "output");

	_network.init();

	BackProp optimizer(&_network);
	optimizer.init(new CrossEntropyCost(), 0.001, 0.9);


	for (int i = 0; i < 100; i++)
	{
		float error = 0;
		auto start = chrono::high_resolution_clock::now();
		for (int s = 0; s < _train_input.size() / 10; s++)
		{
			error += optimizer.train(_train_input[s], _train_target[s]);
		}
		auto finish = chrono::high_resolution_clock::now();
		cout << "MNIST 6000: " << (finish - start).count() * ((float)chrono::high_resolution_clock::period::num / chrono::high_resolution_clock::period::den) << endl;
		cout << error << endl;
	}
}

void CNN::test()
{
	int t = 0;
	int f = 0;

	for (int s = 0; s < _train_input.size(); s++)
	{
		_network.activate(_train_input[s]);
		if (_network.get_output()->max_value_index() == _train_target[s]->max_value_index())
		{
			t++;
		}
		else
		{
			f++;
		}
	}

	cout << "Correct: " << t << endl;
	cout << "Incorrect: " << f << endl;
}
