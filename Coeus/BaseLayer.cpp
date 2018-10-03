#include "BaseLayer.h"
#include "IDGen.h"

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

void BaseLayer::init(vector<BaseLayer*>& p_input_layers)
{
}

Connection* BaseLayer::add_connection(Connection* p_connection) {
	_connections[p_connection->get_id()] = p_connection;

	return p_connection;
}

SimpleCellGroup* BaseLayer::add_group(SimpleCellGroup* p_group) {
	_groups[p_group->get_id()] = p_group;

	return p_group;
}



