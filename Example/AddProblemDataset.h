#pragma once
#include <string>
#include <vector>
#include "Tensor.h"

using namespace std;
using namespace FLAB;

struct AddProblemSequence
{
	Tensor  input;
	Tensor	target;
};

class AddProblemDataset
{
public:

	enum INPUT_TYPE
	{
		INPUT,
		MASK,
		TARGET
	};

	AddProblemDataset();
	~AddProblemDataset();

	void parse_line(string& p_line, INPUT_TYPE p_input);
	void load_data(const string& p_filename);
	vector<AddProblemSequence>* permute();
	vector<AddProblemSequence>* data() { return &_data; }
	pair<vector<Tensor*>, vector<Tensor*>> to_vector();

private:
	void add_item();

	Tensor _input;
	Tensor _mask;
	Tensor _target;
	vector<AddProblemSequence> _data;

};

