#pragma once
#include <string>
#include <vector>
#include "Tensor.h"

using namespace std;
using namespace FLAB;

struct AddProblemSequence
{
	vector<Tensor> input;
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

private:
	void add_item();

	Tensor _input;
	Tensor _mask;
	Tensor _target;
	vector<AddProblemSequence> _data;

};

