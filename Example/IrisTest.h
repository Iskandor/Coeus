#pragma once
#include "LSOM.h"
#include "IrisDataset.h"
#include "LISSOM.h"

using namespace Coeus;

class IrisTest
{
public:
	IrisTest();
	~IrisTest();

	void init();
	void run(int p_epochs);
	void test();

private:
	void save_results(const string p_filename, const int p_dim_x, const int p_dim_y, double* p_data, const int p_category) const;

	LSOM*		_lsom;
	IrisDataset _dataset;
};

