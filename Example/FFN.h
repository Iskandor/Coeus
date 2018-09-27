#pragma once
#include "NeuralNetwork.h"
#include "IrisDataset.h"

using namespace Coeus;

class FFN
{
public:
	FFN();
	~FFN();

	void run();
	void run_iris();

private:
	NeuralNetwork _network;
	IrisDataset _dataset;
};

