#pragma once
#include <string>
#include <bitset>
#include <map>
#include <vector>
#include "Tensor.h"
#include "CisLoader.h"

using namespace FLAB;
using namespace std;

#define BIG_ENDIAN 0
#define LITTLE_ENDIAN 1

struct PackDataRow
{
	bool bought;
	int price_category;
	int region;
	string profiles_register_platform;
	string profiles_register_device;
	string profiles_gender;
	string profiles_country;
	int level;
	double netto;
	int order;
	int login_count;
	int target;

	static bool compare(const PackDataRow &a, const PackDataRow &b)
	{
		return a.order < b.order;
	}
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
	vector<PackDataSequence>* permute();
	vector<PackDataSequence>* data() { return &_data; }
	pair<vector<Tensor*>, vector<Tensor*>> to_vector();



private:
	void parse_line(string& p_line);
	void create_sequence(vector<PackDataRow>& p_row);
	bool has_target(vector<PackDataRow>& p_row) const;

	template<typename T>
	void add_bin_data(T* p_value, vector<double> &p_data) const;
	template<typename T>
	void add_data(T* p_value, int p_size, vector<double> &p_data) const;
	
	template<typename T>
	int* to_binary(T p_value, int p_size = 0) const;

	int get_endian() const;

	map<int, vector<PackDataRow>> *_data_tree;
	vector<PackDataSequence> _data;

	CisLoader _cis_price_category;
	CisLoader _cis_device;
	CisLoader _cis_platform;
	CisLoader _cis_gender;
	CisLoader _cis_country;
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
int* PackDataset::to_binary(T p_value, int p_size) const
{
	const int size = p_size == 0 ? sizeof(T) * 8 : p_size;
	int* binary = new int[size];
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

	return binary;
}