
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

	RNN model;
	//model.run_pack();
	//model.test_pack();
	//model.test_pack_cm();
	//model.test_pack_alt();
	model.run_add_problem();

	/*
	MazeExample example;
	//example.example_q(24, 4e-4, 0, false);

	int c = 0;

	for(int i = 0; i < 10; i++)
	{
		//c += example.example_sarsa(24, 3e-4, 0, false);
		c += example.example_q(48, 1e-4, 0, false);
	}
	cout << c << " / 10" << endl;
	*/

	/*
	Logger::instance().init("maze.log");

	for(int i = 8; i < 9; i++)
	{
		for(int a = -1; a > -6; a--)
		{
			for (int l = 3; l < 4; l++)
			{
				const int hidden = pow(2, i);
				const float alpha = pow(10, a);
				const float lambda = l * 0.2 - 0.1;
				int correct = 0;

				for (int e = 0; e < 10; e++)
				{
					MazeExample example;
					correct += example.example_q(hidden, alpha, lambda, false);
				}

				cout << hidden << ", " << alpha << ", " << lambda << ", " << correct << "/10" << endl;
				Logger::instance().log(to_string(hidden) + ", " + to_string(alpha) + ", " + to_string(lambda) + ", " + to_string(correct) + "/10");

			}

			const int hidden = pow(2, i);
			const float alpha = pow(10, a);
			int correct = 0;
			int c[10];

			parallel_for(0, 10, [&](const int t) {
				MazeExample example;
				c[t] = example.example_q(hidden, alpha, 0, false);
				},
				static_partitioner()
			);

			for (auto ci : c)
			{
				correct += ci;
			}

			cout << hidden << ", " << alpha << ", " << correct << "/10" << endl;
			Logger::instance().log(to_string(hidden) + ", " + to_string(alpha) + ", " + to_string(correct) + "/10");
		}
	}

	Logger::instance().close();
	*/
	/*
	IrisTest iris;

	iris.init();
	iris.run(2000);
	iris.test();
	*/

	system("pause");

	return 0;
}
