#pragma once
#include <string>
#include "Tensor.h"
#include <vector>
#include <map>

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
	static const int CATEGORIES = 3;
	static const int SIZE = 4;

	IrisDataset();
	~IrisDataset();

	void load_data(string p_filename);
	void encode();
	vector<IrisDatasetItem>* permute();

	map<string, int>* get_target_map() { return &_target; }

private:
	void parse_line(string p_line);

	vector<IrisDatasetItem> _data;
	map<string, int> _target;
};

