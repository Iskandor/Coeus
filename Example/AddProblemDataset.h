#pragma once
#include <string>
#include <vector>
#include "Tensor.h"

using namespace std;

struct AddProblemSequence
{
	vector<Tensor*>  input;
	Tensor*			target;
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
	vector<AddProblemSequence>* permute(bool p_batch);
	vector<AddProblemSequence>* data() { return &_data; }
	vector<AddProblemSequence>* raw_data() { return &_raw_data; }
	pair<vector<vector<Tensor*>>, vector<Tensor*>> to_vector();
	void split(int p_batch);

private:
	void add_item();

	Tensor _input;
	Tensor _mask;
	Tensor _target;
	vector<AddProblemSequence> _data;
	vector<AddProblemSequence> _batch_data;
	vector<AddProblemSequence> _raw_data;

};

