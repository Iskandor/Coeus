#pragma once

#include "Maze.h"
#include <NeuralNetwork.h>
#include <vector>
#include <Windows.h>


using namespace Coeus;

class MazeExample
{
public:
	MazeExample();
	~MazeExample();

	int example_q(int p_epochs, bool p_verbose = true);
	void example_double_q(int p_epochs, bool p_verbose = true);
	int example_sarsa(int p_epochs, bool p_verbose = true);
	void example_actor_critic(int p_epochs, bool p_verbose = true);
	void example_nac(int p_epochs, bool p_verbose = true);
	void example_ppo(int p_epochs);
	
	void example_deep_q(int p_epochs, bool p_verbose = true);
	
	void example_a2c(int p_epochs, bool p_verbose = true);
	void example_a3c(int p_epochs, bool p_verbose = true);
	void example_selector(int p_hidden);

private:
	Maze* _maze;
	
	HANDLE _hConsole_c;

	int test_q(NeuralNetwork* p_network, bool p_verbose = true) const;
	void test_v(NeuralNetwork* p_network, bool p_verbose = true) const;
	void test_policy(NeuralNetwork& p_network);

	static Tensor encode_state(vector<float> *p_sensors);
	static int choose_action(Tensor* p_input, float epsilon);
	static void binary_encoding(int p_value, Tensor* p_vector);

	void console_print(string &p_s, int p_x, int p_y);	
	void console_clear();
	TCHAR console_wait();
	string string_format(const std::string fmt_str, ...);
};

