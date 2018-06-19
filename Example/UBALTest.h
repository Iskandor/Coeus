#pragma once

#include "NeuralNetwork.h"

using namespace Coeus;

class UBALTest
{
public:
	UBALTest();
	~UBALTest();

	void run();

private:
	NeuralNetwork _network;
};

