
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


using namespace std;
using namespace MNS;

int main()
{
	/*
	FFN model;

	model.run();
	*/

	RNN model;

	model.run_add_problem();

	/*
	MazeExample example;

	example.example_icm();
	*/

	/*
	IrisTest iris;

	iris.init();
	iris.run(2000);
	iris.test();
	*/

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
