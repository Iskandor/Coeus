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

private:
	static Tensor encode_state(vector<double> *p_sensors);
	static int choose_action(Tensor* p_input, double epsilon);
	static void binary_encoding(double p_value, Tensor* p_vector);
};

