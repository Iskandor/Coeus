#include "IrisDataset.h"
#include <fstream>
#include <vector>
#include <algorithm>

IrisDataset::IrisDataset()
{
}


IrisDataset::~IrisDataset()
{
}

void IrisDataset::load_data(const string p_filename) {
	string line;

	ifstream file(p_filename);

	if (file.is_open())
	{
		while (getline(file, line))
		{
			if (line.size() > 0) parse_line(line);
		}
		file.close();
	}

	Tensor max = Tensor::Value({ 4 }, -1);

	for(int i = 0; i < _data.size(); i++) {
		for(int j = 0; j < 4; j++) {
			if (max.at(j) == -1 || max.at(j) < _data[i].data->at(j)) {
				max.set(j, _data[i].data->at(j));
			}
		}
	}

	for (int i = 0; i < _data.size(); i++) {
		Tensor::apply(_data[i].data, &max, Tensor::ew_div);
	}
}

vector<IrisDatasetItem>* IrisDataset::permute() {
	random_shuffle(_data.begin(), _data.end());
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

	item.data = new Tensor({ 4 }, Tensor::INIT::ZERO);

	for(int i = 0; i < 4; i++) {
		item.data->set(i, stod(tokens[i]));
	}

	item.target = tokens[4];

	_data.push_back(item);
}
