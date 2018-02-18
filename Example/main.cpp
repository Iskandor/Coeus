
#include "Config.h"
#include "Logger.h"
#include "ModelMNS.h"
#include "ModelMNS2.h"
#include "ModelMNS3.h"


using namespace std;
using namespace MNS;

int main()
{
	const string timestamp = to_string(time(nullptr));

	Config::instance().Load("./config3.json");
	Logger::instance().init(timestamp + ".log");

	ModelMNS3 model;

	//model.init();
	//model.run(Config::instance().epoch);
	//model.save(timestamp);

	//model.init("1518009903");
	//model.save_umatrix("1518009903");

	model.init("1518510932");
	//model.testMirror();
	//model.testAllWinners();
	//model.testFinalWinners();
	model.testDistance();
	//model.testBALData();

	Logger::instance().close();

	system("pause");

	return 0;
}
