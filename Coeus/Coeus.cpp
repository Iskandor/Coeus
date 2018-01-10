// Coeus.cpp : Defines the entry point for the console application.
//

#include "../FLAB/Tensor.h"
#include <string>
#include "IDGen.h"
#include "base64.h"
//#include <vld.h>
using namespace FLAB;


int main()
{
	string id1 = Coeus::IDGen::instance().next();
	string id2 = Coeus::IDGen::instance().next();
	string out;

	Base64::Decode(id1, &out);
	Base64::Decode(id2, &out);

    return 0;
}

