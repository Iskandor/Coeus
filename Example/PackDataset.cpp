#include "PackDataset.h"
#include <fstream>
#include <vector>
#include <iostream>
#include "Encoder.h"


using namespace Coeus;

PackDataset::PackDataset()
{
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

	const int player_id = stoi(tokens[1]);

	PackDataRow row;

	row.pack = stoi(tokens[2]);
	row.pack_date = tokens[3];
	row.date = tokens[4];
	row.bought = tokens[5] == "True";
	row.price_category = stoi(tokens[6]);
	row.region = stoi(tokens[7]);

	/*
	0 row_id
	1 player_id                          int64
	2 pack                               int64
	3 pack_date                          object
	4 date                               object
	5 bought                             bool
	6 price_category                     int64
	7 region                             int64
	*/

	row.revenues_lifetime_count = tokens[8].empty() ? 0 : stod(tokens[8]);
	row.revenues_lifetime_sum_netto = tokens[9].empty() ? 0 : stod(tokens[9]);
	row.revenues_lifetime_sum_brutto = tokens[10].empty() ? 0 : stod(tokens[10]);
	row.revenues_lifetime_min_brutto = tokens[11].empty() ? 0 : stod(tokens[11]);
	row.revenues_lifetime_max_brutto = tokens[12].empty() ? 0 : stod(tokens[12]);
	row.revenues_lifetime_first_payment = tokens[13];
	row.revenues_lifetime_last_payment = tokens[14];
	row.revenues_7d_count = tokens[15].empty() ? 0 : stod(tokens[15]);
	row.revenues_7d_sum_brutto = tokens[16].empty() ? 0 : stod(tokens[16]);
	row.revenues_7d_min_brutto = tokens[17].empty() ? 0 : stod(tokens[17]);
	row.revenues_7d_max_brutto = tokens[18].empty() ? 0 : stod(tokens[18]);
	row.revenues_28d_count = tokens[19].empty() ? 0 : stod(tokens[19]);
	row.revenues_28d_sum_brutto = tokens[20].empty() ? 0 : stod(tokens[20]);
	row.revenues_28d_min_brutto = tokens[21].empty() ? 0 : stod(tokens[21]);
	row.revenues_28d_max_brutto = tokens[22].empty() ? 0 : stod(tokens[22]);
	row.revenues_56d_count = tokens[23].empty() ? 0 : stod(tokens[23]);
	row.revenues_56d_sum_brutto = tokens[24].empty() ? 0 : stod(tokens[24]);
	row.revenues_56d_min_brutto = tokens[25].empty() ? 0 : stod(tokens[25]);
	row.revenues_56d_max_brutto = tokens[26].empty() ? 0 : stod(tokens[26]);
	row.revenues_84d_count = tokens[27].empty() ? 0 : stod(tokens[27]);
	row.revenues_84d_sum_netto = tokens[28].empty() ? 0 : stod(tokens[28]);
	row.revenues_84d_sum_brutto = tokens[29].empty() ? 0 : stod(tokens[29]);
	row.revenues_84d_min_brutto = tokens[30].empty() ? 0 : stod(tokens[30]);
	row.revenues_84d_max_brutto = tokens[31].empty() ? 0 : stod(tokens[31]);

	/*
	8 revenues_lifetime_count            float64
	9 revenues_lifetime_sum_netto        float64
	10 revenues_lifetime_sum_brutto       float64
	11 revenues_lifetime_min_brutto       float64
	12 revenues_lifetime_max_brutto       float64
	13 revenues_lifetime_first_payment    datetime64[ns]
	14 revenues_lifetime_last_payment     datetime64[ns]
	15 revenues_7d_count                  float64
	16 revenues_7d_sum_brutto             float64
	17 revenues_7d_min_brutto             float64
	18 revenues_7d_max_brutto             float64
	19 revenues_28d_count                 float64
	20 revenues_28d_sum_brutto            float64
	21 revenues_28d_min_brutto            float64
	22 revenues_28d_max_brutto            float64
	23 revenues_56d_count                 float64
	24 revenues_56d_sum_brutto            float64
	25 revenues_56d_min_brutto            float64
	26 revenues_56d_max_brutto            float64
	27 revenues_84d_count                 float64
	28 revenues_84d_sum_netto             float64
	29 revenues_84d_sum_brutto            float64
	30 revenues_84d_min_brutto            float64
	31 revenues_84d_max_brutto            float64
	*/

	row.activity_7d_login_count = tokens[32].empty() ? 0 : stoi(tokens[32]);
	row.activity_28d_login_count = tokens[33].empty() ? 0 : stoi(tokens[33]);
	row.activity_last_login = tokens[34];
	row.profiles_register_time = tokens[35];
	row.profiles_register_platform = tokens[36];
	row.profiles_register_device = tokens[37];
	row.profiles_birthday = tokens[38];
	row.profiles_gender = tokens[39];
	row.profiles_country = tokens[40];
	row.level = tokens[41].empty() ? 0 : stoi(tokens[41]);

	/*
	32 activity_7d_login_count            float64
	33 activity_28d_login_count           float64
	34 activity_last_login                datetime64[ns]

	35 profiles_register_time             datetime64[ns]
	36 profiles_register_platform         object
	37 profiles_register_device           object
	38 profiles_birthday                  object
	39 profiles_gender                    object
	40 profiles_country                   object
	41 level                              float6*
	 */

	(*_data_tree)[player_id].push_back(row);
}

void PackDataset::create_sequence(vector<PackDataRow>& p_row)
{
	PackDataSequence sequence;

	Tensor input = Tensor::Zero({ static_cast<int>(p_row.size()), 69 });
	Tensor target = Tensor::Zero({ static_cast<int>(p_row.size()), 1 });
	int index = 0;

	for(auto row = p_row.begin(); row != p_row.end(); ++row)
	{
		vector<double> data;

		add_bin_data<int>(to_binary<int>((*row).pack), data);
		add_bin_data<int>(to_binary<int>((*row).price_category), data);

		int* regions = new int[5];
		Encoder::one_hot(regions, 5, (*row).region - 1);
		add_data<int>(regions, 5, data);

		for(int i = 0; i < data.size(); i++)
		{
			input.set(index, i, data[i]);
		}

		target.set(index, 0, (*row).bought ? 1 : 0);
	}

	sequence.input = input;
	sequence.target = target;

	_data.push_back(sequence);
}

void PackDataset::load_data(const string& p_filename)
{
	_data_tree = new map<int, vector<PackDataRow>>;

	string line;

	ifstream file(p_filename);

	if (file.is_open())
	{
		// header
		getline(file, line);

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
		create_sequence(it->second);
	}

	delete _data_tree;
}
