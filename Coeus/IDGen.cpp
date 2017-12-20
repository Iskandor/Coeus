#include "IDGen.h"
#include "base64.h"

using namespace Coeus;
using namespace std;

IDGen& IDGen::instance() {
	static IDGen instance;
	return instance;
}

string IDGen::next() {
	string res;
	Base64::Encode(to_string(_id), &res);
	_id++;

	return res;
}
