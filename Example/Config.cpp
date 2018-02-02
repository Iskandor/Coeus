#include "Config.h"
#include <fstream>

using namespace MNS;

Config& Config::instance()
{
	static Config config;
	return config;
}

void Config::Load(string p_filename)
{
	json data;
	ifstream file;

	try {
		file.open(p_filename);
	}
	catch (std::ios_base::failure& e) {
		std::cerr << e.what() << '\n';
	}

	if (file.is_open()) {
		file >> data;

		epoch = data["epochs"].get<int>();
		settling = data["settling"].get<int>();
		motor_data = data["motor_data"].get<string>();
		visual_data = data["visual_data"].get<string>();
		f5_config = parse_msom_config(data["f5"]);
		sts_config = parse_msom_config(data["sts"]);
		pf_config = parse_som_config(data["pf"]);
	}
	file.close();
}

Config::Config()
{
}


Config::~Config()
{
}

SOM_config Config::parse_som_config(json p_data)
{
	SOM_config config;

	config.dim_x = p_data["dim_x"].get<int>();
	config.dim_y = p_data["dim_y"].get<int>();
	config.alpha = p_data["alpha"].get<double>();

	return config;
}

MSOM_config Config::parse_msom_config(json p_data)
{
	MSOM_config config;

	config.dim_x = p_data["dim_x"].get<int>();
	config.dim_y = p_data["dim_y"].get<int>();
	config.alpha = p_data["alpha"].get<double>();
	config.beta = p_data["beta"].get<double>();
	config.gamma1 = p_data["gamma1"].get<double>();
	config.gamma2 = p_data["gamma2"].get<double>();

	return config;
}
