#pragma once
#include "NeuralNetwork.h"

using namespace Coeus;

class RNN
{
public:
	RNN();
	~RNN();

	void run_add_problem();
	void test_add_problem(NeuralNetwork& p_network) const;
	void run_pack();
	void test_pack_cm() const;
	void test_pack_alt() const;

private:
	json load_config(const string& p_filename) const;
};

