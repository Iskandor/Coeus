
#include "Config.h"
#include "Logger.h"
#include "ModelMNS.h"
#include "ModelMNS2.h"
#include "ModelMNS3.h"
#include "FFN.h"
#include "IrisTest.h"
#include "RNN.h"
#include "MazeExample.h"
#include "Encoder.h"
#include <bitset>


using namespace std;
using namespace MNS;

int main()
{
	//FFN model;
	//model.run();

	RNN model;
	model.run_pack();
	//model.test_pack();
	//model.test_pack_cm();
	//model.run_add_problem();
	/*
	Logger::instance().init("maze.log");

	for(int i = 8; i < 9; i++)
	{
		for(int a = -1; a > -6; a--)
		{
			for (int l = 3; l < 4; l++)
			{
				const int hidden = pow(2, i);
				const double alpha = pow(10, a);
				const double lambda = l * 0.2 - 0.1;
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
			const double alpha = pow(10, a);
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

	//MazeExample example;
	//cout << example.example_q(256, 1e-4, 0, false);

	/*
	const string timestamp = to_string(time(nullptr));

	Config::instance().Load("./config0.json");
	Logger::instance().init(timestamp + ".log");

	ModelMNS model;

	model.init();
	model.run(Config::instance().epoch);
	model.save(timestamp);
	*/

	//model.init("1519198741");
	//model.save_umatrix("1519198741");

	//model.init("1519295153");
	//model.testMirror(3);
	//model.testAllWinners();
	//model.testFinalWinners();
	//model.testDistance();
	//model.testBALData();

	//Logger::instance().close();

	system("pause");

	return 0;
}
