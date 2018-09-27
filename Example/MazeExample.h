#pragma once
#include "Tensor.h"
#include <vector>

using namespace FLAB;

class MazeExample
{
public:
	MazeExample();
	~MazeExample();

	void example_q();
	void example_double_q();
	void example_sarsa();
	void example_actor_critic();
	void example_deep_q();
	void example_icm();

private:
	static Tensor encode_state(vector<double> *p_sensors);
	static int choose_action(Tensor* p_input, double epsilon);
	static void binary_encoding(double p_value, Tensor* p_vector);
};

