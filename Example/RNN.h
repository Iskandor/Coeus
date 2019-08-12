#pragma once
#include "NeuralNetwork.h"

using namespace Coeus;

class RNN
{
public:
	RNN();
	~RNN();

	void run_add_problem();
	void run_sin_prediction();
	void run_add_problem_gru();
	void test_add_problem(NeuralNetwork& p_network) const;
	void run_pack();
	void run_pack2();
	void test_pack_cm() const;
	void test_pack_alt() const;

private:
	json load_config(const string& p_filename) const;
};

