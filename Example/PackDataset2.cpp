#include "PackDataset2.h"
#include <fstream>
#include <vector>
#include <iostream>
#include <Encoder.h>
#include <algorithm>
#include <random>
#include <sstream>


using namespace Coeus;

PackDataset2::PackDataset2()
{
	_cis_country.load_data("./data/cis_country.csv");
	_cis_device.load_data("./data/cis_device.csv");
	_cis_gender.load_data("./data/cis_gender.csv");
	_cis_platform.load_data("./data/cis_platform.csv");
	_cis_price_category.load_data("./data/cis_price_category.csv");
	_cis_region.load_data("./data/cis_region.csv");
	_pack_definition.load("./data/pack.csv");

	_player_id = 0;
}


PackDataset2::~PackDataset2()
{
	for (auto& it : _raw_data)
	{
		for(auto& i : it.input)
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

string PackDataset2::print()
{
	stringstream ss;

	for(auto it = _raw_data.begin(); it != _raw_data.end(); ++it)
	{
		ss << to_string((*it).player_id) << endl;
		for(auto s = (*it).input.begin(); s != (*it).input.end(); ++s)
		{
			ss << **s << endl;
		}
		ss << *(*it).target << endl;
	}

	return ss.str();
}

void PackDataset2::parse_line(string& p_line)
{
	vector<string> tokens;

	size_t pos = 0;

	while ((pos = p_line.find(',')) != std::string::npos) {
		const string token = p_line.substr(0, pos);
		tokens.push_back(token);
		p_line.erase(0, pos + 1);
	}

	tokens.push_back(p_line);

	PackDataRow2 row;

	row.player_id = stoi(tokens[0]);
	row.brutto = stod(tokens[3]);
	row.target = stoi(tokens[4]);
	row.pack_id = stoi(tokens[5]);
	row.bought = stoi(tokens[6]) == 1;
	row.price_category = stoi(tokens[7]);
	row.gems_status = stoi(tokens[8]);
	row.level = stoi(tokens[9]);
	row.login_count = stoi(tokens[10]);
	row.region = stoi(tokens[11]);
	row.profiles_country = tokens[12];
	row.profiles_register_device = tokens[13];
	row.profiles_register_platform = tokens[14];
	row.event_material = stoi(tokens[15]);
	row.usable = stoi(tokens[16]);
	row.material = stoi(tokens[17]);
	row.building = stoi(tokens[18]);
	row.system = stoi(tokens[19]);
	row.decoration = stoi(tokens[20]);
	row.token = stoi(tokens[21]);
	row.skin = stoi(tokens[22]);
	row.order = stoi(tokens[25]);

	_data_tree[row.player_id].push_back(row);
}

Tensor* PackDataset2::encode_row(PackDataRow2& p_row)
{
	vector<Tensor*> value_list;

	Tensor brutto({ 20 }, Tensor::ZERO);
	Encoder::pop_code(brutto, p_row.brutto, 0, 190);
	value_list.push_back(&brutto);

	Tensor price_category({ _cis_price_category.category_count() }, Tensor::ZERO);
	Encoder::one_hot(price_category, _cis_price_category.get_key(to_string(p_row.price_category)));
	value_list.push_back(&price_category);

	int* gems_status_bin = to_binary(p_row.gems_status, 17);
	Tensor gems_status({ 17 }, Tensor::ZERO);
	gems_status.override(gems_status_bin);
	value_list.push_back(&gems_status);

	int* level_bin = to_binary(p_row.level, 10);
	Tensor level({ 10 }, Tensor::ZERO);
	level.override(level_bin);
	value_list.push_back(&level);

	int* login_count_bin = to_binary(p_row.login_count, 11);
	Tensor login_count({ 11 }, Tensor::ZERO);
	login_count.override(login_count_bin);
	value_list.push_back(&login_count);

	Tensor region({ _cis_region.category_count() }, Tensor::ZERO);
	Encoder::one_hot(region, _cis_region.get_key(to_string(p_row.region)));
	value_list.push_back(&region);

	int* country_bin = to_binary(_cis_country.get_key(p_row.profiles_country), 8);
	Tensor country({ 8 }, Tensor::ZERO);
	country.override(country_bin);
	value_list.push_back(&country);

	Tensor device({ _cis_device.category_count() }, Tensor::ZERO);
	Encoder::one_hot(device, _cis_device.get_key(p_row.profiles_register_device));
	value_list.push_back(&device);

	Tensor platform({ _cis_platform.category_count() }, Tensor::ZERO);
	Encoder::one_hot(platform, _cis_platform.get_key(p_row.profiles_register_platform));
	value_list.push_back(&platform);

	int* event_material_bin = to_binary(p_row.event_material, 10);
	Tensor event_material({ 10 }, Tensor::ZERO);
	event_material.override(event_material_bin);
	value_list.push_back(&event_material);

	Tensor usable({ 4 }, Tensor::ZERO);
	Encoder::one_hot(usable, p_row.usable);
	value_list.push_back(&usable);

	Tensor material({ 4 }, Tensor::ZERO);
	Encoder::one_hot(material, p_row.material);
	value_list.push_back(&material);

	Tensor building({ 4 }, Tensor::ZERO);
	Encoder::one_hot(building, p_row.building);
	value_list.push_back(&building);

	Tensor system({ 4 }, Tensor::ZERO);
	Encoder::one_hot(system, p_row.system);
	value_list.push_back(&system);

	Tensor decoration({ 4 }, Tensor::ZERO);
	Encoder::one_hot(decoration, p_row.decoration);
	value_list.push_back(&decoration);

	Tensor token({ 4 }, Tensor::ZERO);
	Encoder::one_hot(token, p_row.token);
	value_list.push_back(&token);

	Tensor skin({ 4 }, Tensor::ZERO);
	Encoder::one_hot(skin, p_row.skin);
	value_list.push_back(&skin);

	return Tensor::concat(value_list);
}

void PackDataset2::create_sequence(vector<PackDataRow2>& p_sequence)
{
	Tensor *target = new Tensor({ 1 }, Tensor::ZERO);


	while(has_target(p_sequence)) {

		PackDataSequence2 sequence;

		int index = -1;
		vector<Tensor*> input_list;

		do		
		{
			index++;
			Tensor* input = encode_row(p_sequence[index]);

			_input_dim = input->size();

			input_list.push_back(input);

			if (p_sequence[index].target == 1)
			{
				(*target)[0] = p_sequence[index].bought ? 1 : 0;
			}
			

		} while (p_sequence[index].target == 0);

		p_sequence[index].target = 0;

		

		for(auto i = 0; i < input_list.size(); i++)
		{
			sequence.input.push_back(input_list[i]);
		}

		sequence.player_id = p_sequence[0].player_id;

		sequence.target = target;
		

		_raw_data.push_back(sequence);
	}

}

void PackDataset2::create_sequence2(vector<PackDataRow2>& p_sequence)
{
	PackDataSequence3 sequence;

	for(PackDataRow2& row : p_sequence)
	{
		Tensor* input = encode_row(row);
		Tensor *target = nullptr;

		_input_dim = input->size();

		if (row.target == 1)
		{
			target = new Tensor({ 1 }, Tensor::ZERO);
			target->set(0, row.bought ? 1 : 0);
		}

		sequence.player_id = row.player_id;
		sequence.input.push_back(input);
		sequence.target.push_back(target);
	}
	_raw_data2.push_back(sequence);
}

vector<PackDataRow2> PackDataset2::get_sequence(const int p_player, const bool p_test)
{
	const int limit = 5;
	int start = 0;
	int start_index = 0;
	int target_index = limit;
	int target_count = 0;
	vector<PackDataRow2> sequence = _data_tree[p_player];
	vector<PackDataRow2> result;

	for (auto& s : sequence)
	{
		if (s.target == 1) target_count++;
	}

	if (p_test) {
		start = target_count - limit;
		target_index = target_count;
	}

	for (int i = start; i < target_count - limit + 1; i++)
	{
		PackDataSequence2 data;
		for (auto& s : sequence)
		{
			if (start_index >= target_index - limit && start_index < target_index)
			{
				result.push_back(s);
			}
			if (s.target == 1)
			{
				start_index++;
			}
			if (start_index == target_index)
			{
				start_index = 0;
				target_index++;
				break;
			}
		}
	}

	return result;
}

void PackDataset2::create_sequence_test(vector<PackDataRow2>& p_sequence)
{
	Tensor *target = new Tensor({ 1 }, Tensor::ZERO);


	PackDataSequence2 sequence;

	int index = -1;
	vector<Tensor*> input_list;

	for(auto it = p_sequence.begin() ; it != p_sequence.end(); ++it) {
		index++;
		Tensor* input = encode_row(p_sequence[index]);

		_input_dim = input->size();

		input_list.push_back(input);

		if (p_sequence[index].target == 1)
		{
			(*target)[0] = p_sequence[index].bought ? 1 : 0;
		}
	}

	sequence.player_id = p_sequence[0].player_id;

	for (auto i = 0; i < input_list.size(); i++)
	{
		sequence.input.push_back(input_list[i]);
	}

	sequence.target = target;

	_raw_data.push_back(sequence);
}

void PackDataset2::create_sequence_solid(vector<PackDataRow2>& p_sequence, const bool p_test)
{
	const int limit = 5;
	int start = 0;
	int start_index = 0;
	int target_index = limit;
	int target_count = 0;

	for (auto& s : p_sequence)
	{
		if (s.target == 1) target_count++;
	}

	if (p_test) {
		start = target_count - limit;
		target_index = target_count;
	}

	for (int i = start; i < target_count - limit + 1; i++)
	{
		PackDataSequence2 data;
		for (auto& s : p_sequence)
		{
			if (start_index >= target_index - limit && start_index < target_index)
			{
				data.pack_id.push_back(s.pack_id);
				data.input.push_back(encode_row(s));
				//cout << s.target << " "  << s.order << endl;
			}
			if (s.target == 1)
			{
				start_index++;
			}
			if (start_index == target_index)
			{
				start_index = 0;
				data.player_id = s.player_id;
				data.target = new Tensor({ 1 }, Tensor::VALUE, s.bought ? 1 : 0);
				_raw_data.push_back(data);
				target_index++;
				break;
			}
		}
	}

}

bool PackDataset2::has_target(vector<PackDataRow2>& p_sequence) const
{
	bool result = false;

	for(auto it = p_sequence.begin(); it != p_sequence.end(); ++it)
	{
		result = result || it->target == 1;
	}

	return result;
}

bool PackDataset2::compare(const PackDataSequence2& x, PackDataSequence2 y)
{
	return x.input.size() > y.input.size();
}

int PackDataset2::get_endian() const
{
	short int word = 0x0001;
	char *b = (char *)&word;
	return (b[0] ? LITTLE_ENDIAN : BIG_ENDIAN);
}

void PackDataset2::load_data(const string& p_filename, bool p_prob, const bool p_test)
{
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

	for (auto it = _data_tree.begin(); it != _data_tree.end(); ++it)
	{
		sort(it->second.begin(), it->second.end(), PackDataRow2::compare);
		create_sequence_solid(it->second, p_test);
		if (p_test)
		{
			//create_sequence_test(it->second);			
		}
		else if (p_prob)
		{
		}
		else
		{
			create_sequence2(it->second);
			//create_sequence(it->second);
		}
		
	}
}

vector<PackDataSequence2>* PackDataset2::permute(const bool p_batch)
{
	vector<PackDataSequence2>* result;

	if (p_batch)
	{
		result = &_batch_data;
	}
	else
	{
		result = &_raw_data;
	}

	shuffle(result->begin(), result->end(), std::mt19937(std::random_device()()));
	return result;
}

vector<PackDataSequence3>* PackDataset2::permute()
{
	shuffle(_raw_data2.begin(), _raw_data2.end(), std::mt19937(std::random_device()()));
	return &_raw_data2;
}

pair<vector<Tensor*>, vector<Tensor*>> PackDataset2::to_vector()
{
	vector<Tensor*> input;
	vector<Tensor*> target;

	/*
	vector<PackDataSequence2>* data = permute();
	//vector<PackDataSequence2>* data = &_data;

	for (auto it = data->begin(); it != data->end(); ++it)
	{
		input.push_back(&(*it).input);
		target.push_back(&(*it).target);
	}
	*/

	return pair<vector<Tensor*>, vector<Tensor*>>(input, target);
}

vector<PackDataSequence2> PackDataset2::create_sequence_test(const int p_player)
{
	vector<PackDataRow2> p_sequence = _data_tree[p_player];
	vector<PackDataSequence2> result;

	for(int c = 0; c < _cis_price_category.category_count(); c++)
	{
		Tensor* input;
		Tensor* target = new Tensor({ 1 }, Tensor::ZERO);

		PackDataSequence2 sequence;
		int index = -1;
		vector<Tensor*> input_list;

		for (auto it = p_sequence.begin(); it != p_sequence.end(); ++it) {
			vector<Tensor*> value_list;
			index++;

			Tensor price_category({ _cis_price_category.category_count() }, Tensor::ZERO);
			if (index == p_sequence.size() - 1)
			{
				Encoder::one_hot(price_category, _cis_price_category.get_data()->at(c).key);
			}
			else
			{
				Encoder::one_hot(price_category, _cis_price_category.get_key(to_string(p_sequence[index].price_category)));
			}
			
			Tensor brutto({ 20 }, Tensor::ZERO);
			Encoder::pop_code(brutto, p_sequence[index].brutto, 0, 150);
			value_list.push_back(&brutto);

			int* gems_status_bin = to_binary(p_sequence[index].gems_status, 16);
			Tensor gems_status({ 16 }, Tensor::ZERO);
			gems_status.override(gems_status_bin);
			value_list.push_back(&gems_status);

			int* level_bin = to_binary(p_sequence[index].level, 16);
			Tensor level({ 16 }, Tensor::ZERO);
			level.override(level_bin);
			value_list.push_back(&level);

			int* login_count_bin = to_binary(p_sequence[index].login_count, 16);
			Tensor login_count({ 16 }, Tensor::ZERO);
			login_count.override(login_count_bin);
			value_list.push_back(&login_count);

			Tensor region({ _cis_region.category_count() }, Tensor::ZERO);
			Encoder::one_hot(region, _cis_region.get_key(to_string(p_sequence[index].region)));
			value_list.push_back(&region);

			Tensor country({ _cis_country.category_count() }, Tensor::ZERO);
			Encoder::one_hot(country, _cis_country.get_key(p_sequence[index].profiles_country));
			value_list.push_back(&country);

			Tensor device({ _cis_device.category_count() }, Tensor::ZERO);
			Encoder::one_hot(device, _cis_device.get_key(p_sequence[index].profiles_register_device));
			value_list.push_back(&device);

			Tensor platform({ _cis_platform.category_count() }, Tensor::ZERO);
			Encoder::one_hot(platform, _cis_platform.get_key(p_sequence[index].profiles_register_platform));
			value_list.push_back(&platform);

			int* event_material_bin = to_binary(p_sequence[index].event_material, 16);
			Tensor event_material({ 16 }, Tensor::ZERO);
			event_material.override(event_material_bin);
			value_list.push_back(&event_material);

			Tensor usable({ 4 }, Tensor::ZERO);
			Encoder::one_hot(usable, p_sequence[index].usable);
			value_list.push_back(&usable);

			Tensor material({ 4 }, Tensor::ZERO);
			Encoder::one_hot(material, p_sequence[index].material);
			value_list.push_back(&material);

			Tensor building({ 4 }, Tensor::ZERO);
			Encoder::one_hot(building, p_sequence[index].building);
			value_list.push_back(&building);

			Tensor system({ 4 }, Tensor::ZERO);
			Encoder::one_hot(system, p_sequence[index].system);
			value_list.push_back(&system);

			Tensor decoration({ 4 }, Tensor::ZERO);
			Encoder::one_hot(decoration, p_sequence[index].decoration);
			value_list.push_back(&decoration);

			Tensor token({ 4 }, Tensor::ZERO);
			Encoder::one_hot(token, p_sequence[index].token);
			value_list.push_back(&token);

			Tensor skin({ 4 }, Tensor::ZERO);
			Encoder::one_hot(skin, p_sequence[index].skin);
			value_list.push_back(&skin);

			input = Tensor::concat(value_list);

			input_list.push_back(input);

			if (p_sequence[index].target == 1)
			{
				(*target)[0] = p_sequence[index].bought ? 1 : 0;
			}
		}

		sequence.player_id = p_sequence[0].player_id;

		for (auto i = 0; i < input_list.size(); i++)
		{
			sequence.input.push_back(input_list[i]);
		}

		sequence.target = target;

		result.push_back(sequence);
	}

	return vector<PackDataSequence2>(result);
}

vector<PackDataSequence2> PackDataset2::get_alt_sequence(const int p_player)
{
	vector<PackDataSequence2> result;

	vector<PackDataRow2> data = get_sequence(p_player, true);
	const int pack_id = data[data.size() - 1].pack_id;
	vector<PackRow> packs = _pack_definition.get_packs(pack_id);

	for(auto pack : packs)
	{
		PackDataSequence2 sequence;
		sequence.player_id = p_player;
		sequence.pack_id.push_back(pack.id);
		for (auto row : data)
		{
			row.pack_id = pack.id;
			row.price_category = pack.category;
			row.building = pack.building;
			row.decoration = pack.decoration;
			row.event_material = pack.event_material;
			row.material = pack.material;
			row.skin = pack.skin;
			row.system = pack.system;
			row.token = pack.token;
			row.usable = pack.usable;

			sequence.input.push_back(encode_row(row));
		}
		result.push_back(sequence);
	}


	return result;
}

void PackDataset2::split(const int p_batch)
{
	sort(_raw_data.begin(), _raw_data.end(), compare);

	const int batch_size = _raw_data.size() / p_batch + (_raw_data.size() % p_batch == 0 ? 0 : 1);
	int index = 0;	
	const int dim = _raw_data[0].input[0]->size();
	Tensor padding({ dim }, Tensor::ZERO);
	//_data.clear();

	for (int i = 0; i < batch_size; i++)
	{
		int max_len = 0;
		PackDataSequence2 data;
		data.target = new Tensor({ p_batch, _raw_data[index].target->size() }, Tensor::ZERO);

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

			for (int j = 0; j < p_batch; j++)
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


int* PackDataset2::to_binary(int p_value, int p_size) const
{
	/*
	const int size = p_size == 0 ? sizeof(T) * 8 : p_size;
	int* binary = static_cast<int*>(calloc(size, sizeof(int)));
	char data[sizeof(T)];
	memcpy(data, &p_value, sizeof p_value);

	int limit = p_size == 0 ? sizeof(T) : p_size / 8;

	for (int i = 0; i < limit; i++)
	{
	const bitset<8> set(data[i]);
	for(int j = 0; j < 8; j++)
	{
	if (get_endian() == BIG_ENDIAN)
	{
	binary[i * 8 + j] = set[j];
	}
	if (get_endian() == LITTLE_ENDIAN)
	{
	binary[i * 8 + j] = set[7 - j];
	}

	}
	}
	*/
	int* binary = static_cast<int*>(calloc(p_size, sizeof(int)));
	int value = p_value;
	int index = p_size - 1;

	while (value > 0)
	{
		binary[index] = value % 2;
		value /= 2;
		index--;
	}

	return binary;
}
