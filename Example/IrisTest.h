#pragma once
#include "LSOM.h"
#include "IrisDataset.h"

using namespace Coeus;

class IrisTest
{
public:
	IrisTest();
	~IrisTest();

	void init();
	void run(int p_epochs);

private:
	LSOM*		_lsom;
	IrisDataset _dataset;
};

