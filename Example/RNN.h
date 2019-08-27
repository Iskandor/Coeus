#pragma once
#include "NeuralNetwork.h"
#include "PackDataset2.h"

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
	void test_pack_cm();
	void test_pack_alt() const;

private:
	float test_performance(NeuralNetwork* p_network, vector<PackDataSequence2>* p_dataset);
	json load_config(const string& p_filename) const;
};

