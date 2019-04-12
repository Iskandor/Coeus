#include "IrisDataset.h"
#include <fstream>
#include <random>
#include <vector>
#include <algorithm>
#include <set>
#include "Encoder.h"
#include "TensorOperator.h"

using namespace Coeus;

IrisDataset::IrisDataset()
{
}


IrisDataset::~IrisDataset()
{
}

void IrisDataset::load_data(const string& p_filename) {
	string line;

	ifstream file(p_filename);

	if (file.is_open())
	{
		while (getline(file, line))
		{
			if (!line.empty()) parse_line(line);
		}
		file.close();
	}

	Tensor max = Tensor::Value({ SIZE }, -1);

	set<string> temp_target;

	for(int i = 0; i < _data.size(); i++) {

		for(int j = 0; j < SIZE; j++) {
			if (max.at(j) == -1 || max.at(j) < _data[i].data->at(j)) {
				max.set(j, _data[i].data->at(j));
			}
		}

		temp_target.insert(_data[i].target);
	}

	for (int i = 0; i < _data.size(); i++) {
		TensorOperator::instance().vv_ewdiv(_data[i].data->arr(), max.arr(), _data[i].data->arr(), max.size());
	}

	int id = 0;

	for(auto it = temp_target.begin(); it != temp_target.end(); ++it) {
		_target[*it] = id;
		id++;
	}
}

void IrisDataset::encode() {
	
	Tensor result({ 32 }, Tensor::ZERO);
	
	for (int i = 0; i < _data.size(); i++) {
		Tensor* val = new Tensor({ result.size() * SIZE }, Tensor::ZERO);

		for (int j = 0; j < SIZE; j++) {
			Encoder::pop_code(result, _data[i].data->at(j));
			val->push_back(&result);
		}

		delete _data[i].data;
		_data[i].data = val;
	}
}

vector<IrisDatasetItem>* IrisDataset::permute() {
	shuffle(_data.begin(), _data.end(), std::mt19937(std::random_device()()));
	return &_data;
}

void IrisDataset::parse_line(string p_line) {
	vector<string> tokens;

	size_t pos = 0;

	while ((pos = p_line.find(",")) != std::string::npos) {
		const string token = p_line.substr(0, pos);
		tokens.push_back(token);
		p_line.erase(0, pos + 1);
	}

	tokens.push_back(p_line);

	IrisDatasetItem item;

	item.data = new Tensor({ SIZE }, Tensor::INIT::ZERO);

	for(int i = 0; i < SIZE; i++) {
		item.data->set(i, stod(tokens[i]));
	}

	item.target = tokens[SIZE];

	_data.push_back(item);
}
