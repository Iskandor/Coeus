#pragma once
#include "NeuralNetwork.h"

using namespace Coeus;

class RNN
{
public:
	RNN();
	~RNN();

	void run();
	void run_add_problem();

private:
	NeuralNetwork _network;
};

