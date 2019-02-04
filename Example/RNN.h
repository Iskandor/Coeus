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
	void run_pack();
	void run_pack2() const;
	void test_add_problem(NeuralNetwork& p_network) const;
	void test_pack() const;
	void test_pack_cm() const;

private:
	json load_config(const string& p_filename) const;
};

