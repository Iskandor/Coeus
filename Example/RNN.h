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
	void test(NeuralNetwork& p_network) const;
};

