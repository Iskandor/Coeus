#include "AddProblemDataset.h"
#include <fstream>
#include <random>
#include <vector>
#include <algorithm>
#include <cassert>


AddProblemDataset::AddProblemDataset()
{
}


AddProblemDataset::~AddProblemDataset()
{
}

void AddProblemDataset::parse_line(string& p_line, INPUT_TYPE p_input)
{
	vector<string> tokens;

	size_t pos = 0;

	while ((pos = p_line.find(',')) != std::string::npos) {
		const string token = p_line.substr(0, pos);
		tokens.push_back(token);
		p_line.erase(0, pos + 1);
	}

	tokens.push_back(p_line);

	switch(p_input)
	{
		case INPUT:
			{
				_input = Tensor::Zero({ static_cast<int>(tokens.size()) });

				for (int i = 0; i < tokens.size(); i++)
				{
					_input[i] = stod(tokens[i]);
				}
			}
			break;
		case MASK:
			{
				_mask = Tensor::Zero({ static_cast<int>(tokens.size()) });

				for (int i = 0; i < tokens.size(); i++)
				{
					_mask[i] = stod(tokens[i]);
				}
			}
			break;
		case TARGET:
			{
				_target = Tensor::Zero({ static_cast<int>(tokens.size()) });

				for (int i = 0; i < tokens.size(); i++)
				{
					_target[i] = stod(tokens[i]);
				}
			}
			break;
		default: ;
	}
}

void AddProblemDataset::load_data(const string& p_filename)
{
	string line;

	ifstream file(p_filename);

	if (file.is_open())
	{
		while (!file.eof())
		{
			getline(file, line);
			if (!line.empty()) parse_line(line, INPUT);
			getline(file, line);
			if (!line.empty()) parse_line(line, MASK);
			getline(file, line);
			if (!line.empty()) parse_line(line, TARGET);

			add_item();
		}
		file.close();
	}
}

vector<AddProblemSequence>* AddProblemDataset::permute()
{
	shuffle(_data.begin(), _data.end(), std::mt19937(std::random_device()()));
	return &_data;
}

pair<vector<Tensor*>, vector<Tensor*>> AddProblemDataset::to_vector()
{
	vector<Tensor*> input;
	vector<Tensor*> target;

	vector<AddProblemSequence>* data = permute();

	for(auto it = data->begin(); it != data->end(); ++it)
	{
		input.push_back(&(*it).input);
		target.push_back(&(*it).target);
	}

	return pair<vector<Tensor*>, vector<Tensor*>>(input, target);
}

void AddProblemDataset::add_item()
{
	AddProblemSequence item;

	if (_input.size() == _mask.size() && _target.size() == 1)
	{
		item.input = Tensor::Zero({ _input.size(), 2 });
		for (int i = 0; i < _input.size(); i++)
		{
			item.input.set(i, 0, _input[i]);
			item.input.set(i, 1, _mask[i]);
		}

		item.target = _target;
	}
	else
	{
		assert(0);
	}

	_data.push_back(item);
}
