#include "BaseLayer.h"
#include "IDGen.h"

using namespace Coeus;

BaseLayer::BaseLayer(const string& p_id): _output_group(nullptr), _input_group(nullptr), _valid(false)
{
	_id = p_id;
}

BaseLayer::BaseLayer(json p_data)
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
	for (auto& _param : _params)
	{
		*_params[_param.first] += p_update[_param.first];
	}
}

json BaseLayer::get_json() const
{
	json data;

	data["id"] = _id;
	data["type"] = _type;

	return data;
}

Connection* BaseLayer::add_connection(Connection* p_connection) {
	_connections[p_connection->get_id()] = p_connection;

	if (p_connection->is_trainable())
	{
		_params[p_connection->get_id()] = p_connection->get_weights();
	}

	return p_connection;
}
