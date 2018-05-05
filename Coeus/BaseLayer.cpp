#include "BaseLayer.h"

using namespace Coeus;

BaseLayer::BaseLayer(const string p_id): _input_group(nullptr), _output_group(nullptr) 
{
	_id = p_id;
	_gradient_component = nullptr;
}

BaseLayer::BaseLayer(nlohmann::json p_data)
{
	_id = p_data["id"].get<string>();
}

BaseLayer::~BaseLayer()
{
}

Connection* BaseLayer::add_connection(Connection* p_connection) {
	_connections[p_connection->get_id()] = p_connection;

	return p_connection;
}



