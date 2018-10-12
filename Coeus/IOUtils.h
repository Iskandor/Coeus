#pragma once
#include "BaseLayer.h"
#include "Connection.h"
#include "SOM.h"
#include "MSOM.h"
#include "json.hpp"
#include "NeuralNetwork.h"

using namespace nlohmann;

namespace Coeus
{

	class __declspec(dllexport) IOUtils
	{
	public:
		IOUtils();
		~IOUtils();

		static void save_network(NeuralNetwork& p_network, const string& p_filename);
		static json load_network(const string& p_filename);
		static BaseLayer* create_layer(json p_data);
		static IActivationFunction* init_activation_function(json p_data);
	private:
		template<typename T>
		static T* create_layer(json p_data);
	};

	template <typename T>
	T* IOUtils::create_layer(json p_data)
	{
		return new T(p_data);
	}
}


