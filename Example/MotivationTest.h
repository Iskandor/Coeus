#pragma once
#include "Maze.h"
#include "NeuralNetwork.h"
#include "ICM.h"

using namespace Coeus;

class MotivationTest
{
public:
	MotivationTest();
	~MotivationTest();

	void test1(int p_episodes);

	void train_icm(ICM& p_icm);
	
private:
	void test_icm_model(ICM& p_icm);
	void test_icm_model2(ICM& p_icm, NeuralNetwork& p_actor);
	void test_v(NeuralNetwork* p_network);
	void test_policy(NeuralNetwork& p_network);
	
	Maze	_maze;
};
