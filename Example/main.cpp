
#include "Config.h"
#include "ModelMNS.h"
#include "ModelMNS2.h"
#include "ModelMNS3.h"


using namespace std;
using namespace MNS;

int main()
{
	Config::instance().Load("./config0.json");

	ModelMNS model;

	//model.init();
	model.init("1518009903");
	//model.run(Config::instance().epoch);
	//model.save();
	//model.save_umatrix("1518009903");

	//model.testMirror();
	//model.testAllWinners();
	//model.testFinalWinners();
	model.testDistance();
	//model.testBALData();

	system("pause");

	return 0;
}
