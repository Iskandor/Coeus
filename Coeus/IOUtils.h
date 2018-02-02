#pragma once
#include "BaseLayer.h"
#include "Connection.h"
#include "SOM.h"
#include "MSOM.h"
#include "json.hpp"

using namespace nlohmann;

namespace Coeus
{

	class __declspec(dllexport) IOUtils
	{
	public:
		IOUtils();
		~IOUtils();

		static json save_layer(BaseLayer* p_layer);
		static BaseLayer* load_layer(json p_data);

		static void save_network(string p_filename, BaseLayer* p_layer);
		static BaseLayer* load_network(string p_filename);

		static NeuralGroup* read_neural_group(json p_data);
		static Connection* read_connection(json p_data);

	private:
		static json write_som(SOM* p_som);
		static json write_msom(MSOM* p_msom);

		static SOM* read_som(json p_data);
		static MSOM* read_msom(json p_data);

		static json write_neural_group(NeuralGroup* p_group);
		static json write_connection(Connection* p_connection);


	};
}


