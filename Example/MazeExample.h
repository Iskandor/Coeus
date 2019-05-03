#pragma once
#include "Tensor.h"
#include <vector>
#include "NeuralNetwork.h"

using namespace Coeus;

class MazeExample
{
public:
	MazeExample();
	~MazeExample();

	int example_q(int p_hidden, float p_alpha, float p_lambda = 0, bool p_verbose = true);
	void example_double_q();
	int example_sarsa(int p_hidden, float p_alpha, float p_lambda = 0, bool p_verbose = true);
	void example_actor_critic(int p_hidden);
	int example_deep_q(int p_hidden, float p_alpha, float p_lambda = 0, bool p_verbose = true);
	void example_icm(int p_hidden);

private:
	int test_q(NeuralNetwork* p_network, bool p_verbose = true) const;
	void test_v(NeuralNetwork* p_network, bool p_verbose = true) const;

	static Tensor encode_state(vector<float> *p_sensors);
	static int choose_action(Tensor* p_input, float epsilon);
	static void binary_encoding(float p_value, Tensor* p_vector);
};

