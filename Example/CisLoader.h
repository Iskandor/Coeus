#pragma once
#include <string>
#include <vector>

using namespace std;

struct CisRow
{
	int		key;
	string	value;
};

class CisLoader
{
public:
	CisLoader();
	~CisLoader();

	void load_data(const string& p_filename);
	int category_count() const { return _data.size(); }
	int get_key(const string& p_value);
	string get_value(int p_key);

private:
	void parse_line(string& p_line);

	vector<CisRow> _data;

};

