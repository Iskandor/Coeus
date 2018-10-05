#include "BaseLayer.h"
#include "IDGen.h"

using namespace Coeus;

BaseLayer::BaseLayer(const string& p_id): _input_group(nullptr), _output_group(nullptr) 
{
	_id = p_id;
}

BaseLayer::BaseLayer(nlohmann::json p_data)
{
	_id = p_data["id"].get<string>();
}

BaseLayer::~BaseLayer()
= default;

void BaseLayer::init(vector<BaseLayer*>& p_input_layers)
{
}

void BaseLayer::update(map<string, Tensor>& p_update)
{
	for (auto it = _params.begin(); it != _params.end(); ++it) {
		*_params[it->first] += p_update[it->first];
	}
}

Connection* BaseLayer::add_connection(Connection* p_connection) {
	_connections[p_connection->get_id()] = p_connection;

	if (p_connection->is_trainable())
	{
		_params[p_connection->get_id()] = p_connection->get_weights();
	}

	return p_connection;
}