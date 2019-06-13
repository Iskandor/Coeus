#pragma once
#include "mnist_reader.hpp"
#include "NeuralNetwork.h"

using namespace Coeus;

class CNN
{
public:
	CNN();
	~CNN();
	void run();

private:
	NeuralNetwork _network;
	mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> _dataset;
};

