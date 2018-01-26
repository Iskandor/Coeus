
//#include <vld.h>
#include "ModelMNS.h"
#include "ModelMNS2.h"

using namespace std;

int main()
{
	MNS::ModelMNS model;

	model.init();
	model.run(200);
	model.save();

	//model.load("1515883986");
	//model.testMirror();
	//model.testAllWinners();
	//model.testFinalWinners();
	//model.testDistance();
	//model.testBALData();

	return 0;
}
