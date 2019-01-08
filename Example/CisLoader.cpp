#include "CisLoader.h"
#include <fstream>


CisLoader::CisLoader()
{
}


CisLoader::~CisLoader()
{
}

void CisLoader::load_data(const string& p_filename)
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
}

int CisLoader::get_key(const string& p_value)
{
	int result = -1;
	for(auto it = _data.begin(); it != _data.end(); ++it)
	{
		if (it->value == p_value)
		{
			result = it->key;
		}
	}

	return result;
}

string CisLoader::get_value(const int p_key)
{
	string result;
	for (auto it = _data.begin(); it != _data.end(); ++it)
	{
		if (it->key == p_key)
		{
			result = it->value;
		}
	}

	return result;
}

void CisLoader::parse_line(string& p_line)
{
	CisRow row;

	vector<string> tokens;

	size_t pos = 0;

	while ((pos = p_line.find(',')) != std::string::npos) {
		const string token = p_line.substr(0, pos);
		tokens.push_back(token);
		p_line.erase(0, pos + 1);
	}

	tokens.push_back(p_line);

	row.key = stoi(tokens[0]);
	row.value = tokens[1];

	_data.push_back(row);
}
