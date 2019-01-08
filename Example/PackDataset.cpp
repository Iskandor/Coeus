#include "PackDataset.h"
#include <fstream>
#include <vector>
#include <iostream>
#include "Encoder.h"
#include <algorithm>
#include <random>


using namespace Coeus;

PackDataset::PackDataset()
{
	_cis_country.load_data("./data/cis_country.csv");
	_cis_device.load_data("./data/cis_device.csv");
	_cis_gender.load_data("./data/cis_gender.csv");
	_cis_platform.load_data("./data/cis_platform.csv");
	_cis_price_category.load_data("./data/cis_price_category.csv");	
}


PackDataset::~PackDataset()
{
}

void PackDataset::parse_line(string& p_line)
{
	vector<string> tokens;

	size_t pos = 0;

	while ((pos = p_line.find(',')) != std::string::npos) {
		const string token = p_line.substr(0, pos);
		tokens.push_back(token);
		p_line.erase(0, pos + 1);
	}

	tokens.push_back(p_line);

	const int player_id = stoi(tokens[0]);

	PackDataRow row;

	row.bought = tokens[1] == "True";
	row.price_category = stoi(tokens[2]);
	row.region = stoi(tokens[3]);
	row.profiles_register_device = tokens[4];
	row.profiles_register_platform = tokens[5];	
	row.profiles_gender = tokens[6];
	row.profiles_country = tokens[7];
	row.level = stoi(tokens[8]);
	row.netto = stod(tokens[9]);
	row.order = stoi(tokens[10]);
	row.login_count = stoi(tokens[11]);
	row.target = stoi(tokens[12]);

	(*_data_tree)[player_id].push_back(row);
}

void PackDataset::create_sequence(vector<PackDataRow>& p_sequence)
{

	Tensor input;
	Tensor target = Tensor::Zero({ 1 });


	while(has_target(p_sequence)) {

		PackDataSequence sequence;

		int index = -1;
		vector<Tensor> input_list;

		do		
		{
			vector<Tensor> value_list;
			index++;

			Tensor price_category({ _cis_price_category.category_count() }, Tensor::ZERO);
			Encoder::one_hot(price_category, _cis_price_category.get_key(to_string(p_sequence[index].price_category)));
			value_list.push_back(price_category);

			Tensor region({ 5 }, Tensor::ZERO);
			Encoder::one_hot(region, p_sequence[index].region - 1);
			value_list.push_back(region);

			Tensor device({ _cis_device.category_count() }, Tensor::ZERO);
			if (!p_sequence[index].profiles_register_device.empty())
			{
				Encoder::one_hot(device, _cis_device.get_key(p_sequence[index].profiles_register_device));
			}			
			value_list.push_back(device);

			Tensor platform({ _cis_platform.category_count() }, Tensor::ZERO);
			if (!p_sequence[index].profiles_register_platform.empty())
			{
				Encoder::one_hot(platform, _cis_platform.get_key(p_sequence[index].profiles_register_platform));
			}			
			value_list.push_back(platform);

			Tensor gender({ _cis_gender.category_count() }, Tensor::ZERO);
			if (!p_sequence[index].profiles_gender.empty())
			{
				Encoder::one_hot(gender, _cis_gender.get_key(p_sequence[index].profiles_gender));
			}
			value_list.push_back(gender);

			Tensor country({ _cis_country.category_count() }, Tensor::ZERO);
			if (!p_sequence[index].profiles_country.empty())
			{
				Encoder::one_hot(country, _cis_country.get_key(p_sequence[index].profiles_country));
			}
			value_list.push_back(country);

			int* level_bin = to_binary<int>(p_sequence[index].level, 16);
			Tensor level({ 16 }, Tensor::ZERO);
			level.override(level_bin);
			value_list.push_back(level);

			Tensor netto({ 20 }, Tensor::ZERO);
			Encoder::pop_code(netto, p_sequence[index].netto, 0, 100);
			value_list.push_back(netto);

			int* login_count_bin = to_binary<int>(p_sequence[index].login_count, 16);
			Tensor login_count({ 16 }, Tensor::ZERO);
			login_count.override(login_count_bin);
			value_list.push_back(login_count);


			input = Tensor::concat(value_list);

			input_list.push_back(input);

			if (p_sequence[index].target == 1)
			{
				target[0] = p_sequence[index].bought ? 1 : 0;
			}
			

		} while (p_sequence[index].target == 0);

		p_sequence[index].target = 0;

		sequence.input = Tensor::Zero({int(input_list.size()), input.size()});

		for(auto i = 0; i < input_list.size(); i++)
		{
			sequence.input.set_row(input_list[i], i);
		}

		sequence.target = target;

		_data.push_back(sequence);
	}

}

bool PackDataset::has_target(vector<PackDataRow>& p_sequence) const
{
	bool result = false;

	for(auto it = p_sequence.begin(); it != p_sequence.end(); ++it)
	{
		result = result || it->target == 1;
	}

	return result;
}

int PackDataset::get_endian() const
{
	short int word = 0x0001;
	char *b = (char *)&word;
	return (b[0] ? LITTLE_ENDIAN : BIG_ENDIAN);
}

void PackDataset::load_data(const string& p_filename)
{
	_data_tree = new map<int, vector<PackDataRow>>;

	string line;

	ifstream file(p_filename);

	if (file.is_open())
	{
		// data
		while (!file.eof())
		{
			getline(file, line);
			if (!line.empty()) parse_line(line);
		}
		file.close();
	}

	for (auto it = _data_tree->begin(); it != _data_tree->end(); ++it)
	{
		sort(it->second.begin(), it->second.end(), PackDataRow::compare); 
		create_sequence(it->second);
	}

	delete _data_tree;
}

vector<PackDataSequence>* PackDataset::permute()
{
	shuffle(_data.begin(), _data.end(), std::mt19937(std::random_device()()));
	return &_data;
}

pair<vector<Tensor*>, vector<Tensor*>> PackDataset::to_vector()
{
	vector<Tensor*> input;
	vector<Tensor*> target;

	vector<PackDataSequence>* data = permute();
	//vector<PackDataSequence>* data = &_data;

	for (auto it = data->begin(); it != data->end(); ++it)
	{
		input.push_back(&(*it).input);
		target.push_back(&(*it).target);
	}

	return pair<vector<Tensor*>, vector<Tensor*>>(input, target);
}