#include "AddProblemDataset.h"
#include <fstream>
#include <random>
#include <vector>
#include <algorithm>
#include <cassert>
#include <iostream>


AddProblemDataset::AddProblemDataset()
{
}


AddProblemDataset::~AddProblemDataset()
{
	for (auto& it : _raw_data)
	{
		for (auto& i : it.input)
		{
			delete i;
		}
		delete it.target;
	}

	for (auto& it : _batch_data)
	{
		for (auto& i : it.input)
		{
			delete i;
		}
		delete it.target;
	}
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

	int i = 0;

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
			i++;
			//if (i == 10) break;
		}
		file.close();
	}

	int max = 0;

	for (auto& it : _raw_data)
	{
		if (max < it.input.size())
		{
			max = it.input.size();
		}
	}

	Tensor* padding = new Tensor({ _raw_data[0].input[0]->size() }, Tensor::ZERO);

	for (auto& it : _raw_data)
	{
		AddProblemSequence sequence;
		//sequence.input = it.input;
		sequence.target = it.target;

		for(int i = 0; i < max; i++)
		{
			if (i < max - it.input.size())
			{
				sequence.input.push_back(padding);
			}
			else
			{
				sequence.input.push_back(it.input[i - (max - it.input.size())]);
			}
		}

		_data.push_back(sequence);
	}
}

vector<AddProblemSequence>* AddProblemDataset::permute(const bool p_batch)
{
	vector<AddProblemSequence>* result;

	if (p_batch)
	{
		result = &_batch_data;
	}
	else
	{
		result = &_data;
	}

	shuffle(result->begin(), result->end(), std::mt19937(std::random_device()()));
	return result;
}

pair<vector<vector<Tensor*>>, vector<Tensor*>> AddProblemDataset::to_vector()
{
	vector<vector<Tensor*>> input;
	vector<Tensor*> target;

	vector<AddProblemSequence>* data = permute(false);

	for(auto it = data->begin(); it != data->end(); ++it)
	{
		input.push_back((*it).input);
		target.push_back((*it).target);
	}

	return pair<vector<vector<Tensor*>>, vector<Tensor*>>(input, target);
}

void AddProblemDataset::split(const int p_batch)
{
	const int batch_size = _raw_data.size() / p_batch + (_raw_data.size() % p_batch == 0 ? 0 : 1);
	int index = 0;
	int max_len = 0;
	const int dim = _raw_data[0].input[0]->size();
	Tensor padding({ dim }, Tensor::ZERO);
	//_data.clear();

	for(int i = 0; i < batch_size; i++)
	{
		AddProblemSequence data;
		data.target = new Tensor({p_batch, _raw_data[index].target->size()}, Tensor::ZERO);

		index = i * p_batch;

		for (int j = 0; j < p_batch; j++)
		{
			if (max_len < _raw_data[index].input.size())
			{
				max_len = _raw_data[index].input.size();
			}

			data.target->push_back(_raw_data[index].target);

			index++;
			if (index == _raw_data.size())
			{
				index = 0;
			}
		}

		data.input.reserve(max_len);
		for (int k = 0; k < max_len; k++)
		{
			data.input.push_back(new Tensor({ p_batch, dim }, Tensor::ZERO));
		}

		for (int k = 0; k < max_len; k++)
		{			
			index = i * p_batch;			

			for(int j = 0; j < p_batch; j++)
			{
				const int seq_len = _raw_data[index].input.size();

				if (k + seq_len < max_len)
				{
					data.input[k]->push_back(&padding);
				}
				else
				{
					data.input[k]->push_back(_raw_data[index].input[k - (max_len - seq_len)]);
				}

				index++;
				if (index == _raw_data.size())
				{
					index = 0;
				}
			}
		}

		_batch_data.push_back(data);
	}
}

void AddProblemDataset::add_item()
{
	AddProblemSequence item;

	if (_input.size() == _mask.size() && _target.size() == 1)
	{
		
		for (int i = 0; i < _input.size(); i++)
		{
			Tensor* t = new Tensor({ 2 }, Tensor::ZERO);

			t->set(0, _input[i]);
			t->set(1, _mask[i]);
			item.input.push_back(t);
		}

		item.target = new Tensor(_target);
	}
	else
	{
		assert(0);
	}

	_raw_data.push_back(item);
}
