
//#include <vld.h>
#include "ModelMNS.h"
#include "ModelMNS2.h"

using namespace std;

int main()
{
	MNS::ModelMNS2 model;

	model.init();
	model.run(2000);
	//model.save();

	//model.load("1515883986");
	//model.testAllWinners();
	//model.testFinalWinners();
	//model.testDistance();
	//model.testBALData();

	return 0;
}
