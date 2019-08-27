#pragma once
#include <string>
#include <bitset>
#include <map>
#include <vector>
#include "Tensor.h"
#include "CisLoader.h"
#include "PackDefinition.h"

using namespace std;

#define BIG_ENDIAN 0
#define LITTLE_ENDIAN 1

struct PackDataRow2
{
	int player_id;
	int pack_id;
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
	vector<int>		 pack_id;
	vector<Tensor*>  input;
	Tensor*			target;
};

struct PackDataSequence3
{
	int player_id;
	vector<Tensor*> input;
	vector<Tensor*>	target;
};

class PackDataset2
{
public:
	PackDataset2();
	~PackDataset2();

	
	void load_data(const string& p_filename, bool p_prob = false, bool p_test = false);
	vector<PackDataSequence2>* permute(bool p_batch);
	vector<PackDataSequence3>* permute();
	vector<PackDataSequence2>* data() { return &_raw_data; }
	pair<vector<Tensor*>, vector<Tensor*>> to_vector();

	vector<PackDataSequence2> create_sequence_test(int p_player);
	vector<PackDataSequence2> get_alt_sequence(int p_player);

	void split(int p_batch);
	int get_input_dim() const { return _input_dim; }

	string print();

private:
	void parse_line(string& p_line);
	Tensor* encode_row(PackDataRow2& p_row);
	void create_sequence(vector<PackDataRow2>& p_sequence);
	void create_sequence_solid(vector<PackDataRow2>& p_sequence, bool p_test = false);
	void create_sequence_test(vector<PackDataRow2>& p_sequence);
	void create_sequence2(vector<PackDataRow2>& p_sequence);
	vector<PackDataRow2> get_sequence(int p_player, bool p_test = true);
	
	bool has_target(vector<PackDataRow2>& p_row) const;

	static bool compare(const PackDataSequence2& x, PackDataSequence2 y);

	template<typename T>
	void add_bin_data(T* p_value, vector<float> &p_data) const;
	template<typename T>
	void add_data(T* p_value, int p_size, vector<float> &p_data) const;
	
	int* to_binary(int p_value, int p_size = 0) const;

	int get_endian() const;

	map<int, vector<PackDataRow2>> _data_tree;
	vector<PackDataSequence2> _raw_data;
	vector<PackDataSequence3> _raw_data2;
	vector<PackDataSequence2> _batch_data;

	CisLoader _cis_price_category;
	CisLoader _cis_device;
	CisLoader _cis_platform;
	CisLoader _cis_gender;
	CisLoader _cis_country;
	CisLoader _cis_region;
	PackDefinition _pack_definition;

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