
#include "ModelMNS.h"

int main()
{
	MNS::ModelMNS model;

	model.init();
	model.run(2000);
	//model.save();

	//model.load("1512520738");
	model.testFinalWinners();
	//model.testDistance();
	//model.testBALData();

	return 0;
}
