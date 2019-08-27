#pragma once
#include <string>
#include <map>
#include <vector>

using namespace std;

struct PackRow
{
	int id;
	string date;
	int category;
	int event_material;
	int region;
	int usable;
	int material;
	int building;
	int system;
	int decoration;
	int token;
	int skin;
};

class PackDefinition
{
public:
	PackDefinition();
	~PackDefinition();

	void load(const string& p_filename);

	vector<PackRow> get_packs(int p_id);

private:
	void parse_line(string& p_line);
	map<int, PackRow> _data;
};

