#pragma once
#include "Tensor.h"
#include <vector>
#include "NeuralNetwork.h"

using namespace FLAB;
using namespace Coeus;

class MazeExample
{
public:
	MazeExample();
	~MazeExample();

	int example_q(int p_hidden, double p_alpha, double p_lambda = 0, bool p_verbose = true);
	void example_double_q();
	void example_sarsa(bool p_verbose = true);
	void example_actor_critic();
	void example_deep_q();
	void example_icm();

private:
	int test(NeuralNetwork* p_network, bool p_verbose = true) const;

	static Tensor encode_state(vector<double> *p_sensors);
	static int choose_action(Tensor* p_input, double epsilon);
	static void binary_encoding(double p_value, Tensor* p_vector);
};

