#pragma once
#include "NeuralNetwork.h"

using namespace Coeus;

class RNN
{
public:
	RNN();
	~RNN();

	void run();

private:
	Coeus::NeuralNetwork _network;
};

