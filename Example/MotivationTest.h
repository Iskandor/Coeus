#pragma once
#include "Maze.h"
#include "NeuralNetwork.h"
#include "ICM.h"
#include "GM2.h"

using namespace Coeus;

class MotivationTest
{
public:
	MotivationTest();
	~MotivationTest();

	void cart_pole_icm(int p_episodes);
	
	void test_icm(int p_episodes);
	void test_gm2(int p_episodes);

	void train_icm(ICM& p_icm);
	void train_gm2(GM2& p_gm2);
	
private:
	void test_icm_model(ICM& p_icm);
	void test_gm2_model(GM2& p_gm2) const;
	void test_icm_model2(ICM& p_icm, NeuralNetwork& p_actor);
	void test_v(NeuralNetwork* p_network);
	void test_policy(NeuralNetwork& p_network);

	bool evaluate_cart_pole(float p_reward);
	void copy_state(vector<float>& p_observation, Tensor& p_state);
	
	Maze	_maze;
	
	vector<float> _rewards;
};
