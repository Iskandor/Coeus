
#include "Config.h"
#include "ModelMNS.h"
#include "ModelMNS2.h"
#include "ModelMNS3.h"


using namespace std;
using namespace MNS;

int main()
{
	Config::instance().Load("../data/config3.json");

	ModelMNS3 model;

	model.init();
	//model.init("1517840926");
	model.run(Config::instance().epoch);
	model.save();

	//model.testMirror();
	//model.testAllWinners();
	//model.testFinalWinners();
	//model.testDistance();
	//model.testBALData();

	system("pause");

	return 0;
}
