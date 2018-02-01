
//#include <vld.h>
#include "ModelMNS.h"
#include "ModelMNS2.h"
#include "ModelMNS3.h"


using namespace std;

int main()
{
	MNS::ModelMNS3 model;

	model.init();
	model.run(5);
	model.save();

	//model.load("1515883986");
	//model.testMirror();
	//model.testAllWinners();
	//model.testFinalWinners();
	//model.testDistance();
	//model.testBALData();

	return 0;
}
