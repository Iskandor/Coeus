#pragma once
#include "NeuralNetwork.h"

using namespace Coeus;

class FFN
{
public:
	FFN();
	~FFN();

	void run();

private:
	NeuralNetwork _network;
};

