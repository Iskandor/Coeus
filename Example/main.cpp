
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

	Config::instance().Load("./config2.json");
	Logger::instance().init(timestamp + ".log");

	ModelMNS2 model;

	model.init();
	model.run(Config::instance().epoch);
	model.save(timestamp);

	//model.init("1519295153");
	//model.save_umatrix("1519312835");

	//model.init("1519295153");
	//model.testMirror(3);
	//model.testAllWinners();
	//model.testFinalWinners();
	//model.testDistance();
	//model.testBALData();

	Logger::instance().close();

	system("pause");

	return 0;
}
