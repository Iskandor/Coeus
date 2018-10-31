#pragma once
#include <string>
#include <bitset>
#include <map>
#include <vector>
#include "Tensor.h"

using namespace FLAB;
using namespace std;

struct PackDataRow
{
	int pack;
	string pack_date;
	string date;
	bool bought;
	int price_category;
	int region;

	double revenues_lifetime_count;
	double revenues_lifetime_sum_netto;
	double revenues_lifetime_sum_brutto;
	double revenues_lifetime_min_brutto;
	double revenues_lifetime_max_brutto;
	string revenues_lifetime_first_payment;
	string revenues_lifetime_last_payment;
	double revenues_7d_count;
	double revenues_7d_sum_brutto;
	double revenues_7d_min_brutto;
	double revenues_7d_max_brutto;
	double revenues_28d_count;
	double revenues_28d_sum_brutto;
	double revenues_28d_min_brutto;
	double revenues_28d_max_brutto;
	double revenues_56d_count;
	double revenues_56d_sum_brutto;
	double revenues_56d_min_brutto;
	double revenues_56d_max_brutto;
	double revenues_84d_count;
	double revenues_84d_sum_netto;
	double revenues_84d_sum_brutto;
	double revenues_84d_min_brutto;
	double revenues_84d_max_brutto;

	int activity_7d_login_count;
	int activity_28d_login_count;
	string activity_last_login;
	string profiles_register_time;
	string profiles_register_platform;
	string profiles_register_device;
	string profiles_birthday;
	string profiles_gender;
	string profiles_country;
	int level;
};

struct PackDataSequence
{
	Tensor  input;
	Tensor	target;
};

class PackDataset
{
public:
	PackDataset();
	~PackDataset();

	
	void load_data(const string& p_filename);



private:
	void parse_line(string& p_line);
	void create_sequence(vector<PackDataRow>& p_row);

	template<typename T>
	void add_bin_data(T* p_value, vector<double> &p_data) const;
	template<typename T>
	void add_data(T* p_value, int p_size, vector<double> &p_data) const;

	template<typename T>
	int* to_binary(T p_value) const;

	map<int, vector<PackDataRow>> *_data_tree;
	vector<PackDataSequence> _data;
};

template <typename T>
void PackDataset::add_bin_data(T* p_value, vector<double>& p_data) const
{
	for (int i = 0; i < sizeof(T) * 8; i++)
	{
		p_data.push_back(p_value[i]);
	}
}

template <typename T>
void PackDataset::add_data(T* p_value, int p_size, vector<double>& p_data) const
{
	for (int i = 0; i < p_size; i++)
	{
		p_data.push_back(p_value[i]);
	}
}

template <typename T>
int* PackDataset::to_binary(T p_value) const
{
	int* binary = new int[sizeof(T)*8];
	char data[sizeof(T)];
	memcpy(data, &p_value, sizeof p_value);

	for (int i = 0; i < sizeof(T); i++)
	{
		const bitset<8> set(data[i]);
		for(int j = 0; j < 8; j++)
		{
			binary[i * 8 + j] = set[j];
		}
	}

	return binary;
}

