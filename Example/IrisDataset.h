#pragma once
#include <string>
#include "Tensor.h"
#include <vector>

using namespace std;
using namespace FLAB;

struct IrisDatasetItem
{
	Tensor*	data;
	string	target;
};

class IrisDataset
{
public:
	IrisDataset();
	~IrisDataset();

	void load_data(string p_filename);
	vector<IrisDatasetItem>* permute();

private:
	void parse_line(string p_line);

	vector<IrisDatasetItem> _data;	
};

