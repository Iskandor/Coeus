#pragma once

#include <string>
#include <json.hpp>


using namespace nlohmann;
using namespace std;

namespace MNS {

struct SOM_config {
	int dim_x;
	int dim_y;

	float alpha;
};

struct MSOM_config {
	int dim_x;
	int dim_y;

	float alpha;
	float beta;
	float gamma1;
	float gamma2;
};

class Config
{
public:
	static Config& instance();

	void Load(string p_filename);

	int epoch;
	int settling;
	string motor_data;
	string visual_data;
	SOM_config pf_config;
	MSOM_config f5_config;
	MSOM_config sts_config;

private:
	Config();
	~Config();

	SOM_config parse_som_config(json p_data);
	MSOM_config parse_msom_config(json p_data);

};

}

