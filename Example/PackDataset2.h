#pragma once
#include <string>
#include <bitset>
#include <map>
#include <vector>
#include "Tensor.h"
#include "CisLoader.h"

using namespace std;

#define BIG_ENDIAN 0
#define LITTLE_ENDIAN 1

struct PackDataRow2
{
	int player_id;
	float brutto;
	int target;
	bool bought;
	int price_category;
	int gems_status;
	int level;
	int login_count;
	int region;
	string profiles_country;
	string profiles_register_device;
	string profiles_register_platform;
	int event_material;
	int usable;
	int material;
	int building;
	int system;
	int decoration;
	int token;
	int skin;
	int order;

	static bool compare(const PackDataRow2 &a, const PackDataRow2 &b)
	{
		return a.order < b.order;
	}
};

struct PackDataSequence2
{
	int player_id;
	vector<Tensor*>  input;
	Tensor*			target;
};

class PackDataset2
{
public:
	PackDataset2();
	~PackDataset2();

	
	void load_data(const string& p_filename, bool p_prob = false, bool p_test = false);
	vector<PackDataSequence2>* permute(bool p_batch);
	vector<PackDataSequence2>* data() { return &_raw_data; }
	pair<vector<Tensor*>, vector<Tensor*>> to_vector();

	vector<PackDataSequence2> create_sequence_test(int p_player);

	void split(int p_batch);
	int get_input_dim() const { return _input_dim; }

private:
	void parse_line(string& p_line);
	Tensor* encode_row(PackDataRow2& p_row);
	void create_sequence(vector<PackDataRow2>& p_sequence);
	void create_sequence_prob(vector<PackDataRow2>& p_sequence);
	void create_sequence_test(vector<PackDataRow2>& p_sequence);
	
	bool has_target(vector<PackDataRow2>& p_row) const;

	static bool compare(const PackDataSequence2& x, PackDataSequence2 y);

	template<typename T>
	void add_bin_data(T* p_value, vector<float> &p_data) const;
	template<typename T>
	void add_data(T* p_value, int p_size, vector<float> &p_data) const;
	
	template<typename T>
	int* to_binary(T p_value, int p_size = 0) const;

	int get_endian() const;

	map<int, vector<PackDataRow2>> *_data_tree;
	vector<PackDataSequence2> _raw_data;
	vector<PackDataSequence2> _batch_data;

	CisLoader _cis_price_category;
	CisLoader _cis_device;
	CisLoader _cis_platform;
	CisLoader _cis_gender;
	CisLoader _cis_country;
	CisLoader _cis_region;

	int _order;
	int _player_id;
	int _input_dim;
};

template <typename T>
void PackDataset2::add_bin_data(T* p_value, vector<float>& p_data) const
{
	for (int i = 0; i < sizeof(T) * 8; i++)
	{
		p_data.push_back(p_value[i]);
	}
}

template <typename T>
void PackDataset2::add_data(T* p_value, int p_size, vector<float>& p_data) const
{
	for (int i = 0; i < p_size; i++)
	{
		p_data.push_back(p_value[i]);
	}
}

template <typename T>
int* PackDataset2::to_binary(T p_value, int p_size) const
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