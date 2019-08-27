#include "PackDefinition.h"
#include <fstream>
#include <vector>


PackDefinition::PackDefinition()
= default;


PackDefinition::~PackDefinition()
= default;

void PackDefinition::load(const string& p_filename)
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

vector<PackRow> PackDefinition::get_packs(const int p_id)
{
	vector<PackRow> result;

	const string date = _data[p_id].date;
	const int region = _data[p_id].region;

	for(const auto &row : _data)
	{
		if (row.second.date == date && row.second.region == region)
		{
			result.push_back(row.second);
		}
	}

	return result;
}

void PackDefinition::parse_line(string& p_line)
{
	vector<string> tokens;

	size_t pos = 0;

	while ((pos = p_line.find(',')) != std::string::npos) {
		const string token = p_line.substr(0, pos);
		tokens.push_back(token);
		p_line.erase(0, pos + 1);
	}

	tokens.push_back(p_line);

	PackRow row;

	row.id = stoi(tokens[0]);
	row.date = tokens[1];
	row.category = stoi(tokens[2]);
	row.event_material = stoi(tokens[3]);
	row.region = stoi(tokens[4]);
	row.usable = stoi(tokens[5]);
	row.material = stoi(tokens[6]);
	row.building = stoi(tokens[7]);
	row.system = stoi(tokens[8]);
	row.decoration = stoi(tokens[9]);
	row.token = stoi(tokens[10]);
	row.skin = stoi(tokens[11]);

	_data[row.id] = row;
}
