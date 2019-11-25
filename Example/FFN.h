#pragma once
#include "IrisDataset.h"
#include <NeuralNetwork.h>

using namespace Coeus;

class FFN
{
public:
	FFN();
	~FFN();

	void run();
	void run_iris();

private:
	IrisDataset _dataset;
};

