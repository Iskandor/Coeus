
#include "ModelMNS.h"
#include <thread>

using namespace std;

int main()
{
	MNS::ModelMNS model;

	model.init();
	//model.run(2000);
	//model.save();

	model.load("1515267890");
	model.save();
	//model.testFinalWinners();
	//model.testDistance();
	//model.testBALData();

	return 0;
}