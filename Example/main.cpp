
#include "FFN.h"
#include "IrisTest.h"
#include "RNN.h"
#include "MazeExample.h"
#include "Encoder.h"
#include <bitset>


using namespace std;

int main()
{
	//FFN model;
	//model.run();
	//model.run_iris();

	//RNN model;
	//model.run_pack();
	//model.test_pack();
	//model.test_pack_cm();
	//model.test_pack_alt();
	//model.run_add_problem();
	
	MazeExample example;
	//example.example_q(64, 1e-3, 0, true);
	//example.example_sarsa(64, 1e-3, 0, true);
	example.example_actor_critic(64);
	//example.example_deep_q(64, 1e-3, 0, true);
	//example.example_icm(64);

	/*
	IrisTest iris;

	iris.init();
	iris.run(2000);
	iris.test();
	*/

	system("pause");

	return 0;
}
