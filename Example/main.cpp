
#include <IDGen.h>
#include <base64.h>

using namespace std;
using namespace Coeus;

int main()
{
	string id1 = IDGen::instance().next();
	string id2 = IDGen::instance().next();
	string out;

	Base64::Decode(id1, &out);
	Base64::Decode(id2, &out);

	return 0;
}