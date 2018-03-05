#include "BaseLayer.h"

using namespace Coeus;

BaseLayer::BaseLayer(string p_id): _input_group(nullptr), _output_group(nullptr) 
{
	_id = p_id;
}

BaseLayer::BaseLayer(nlohmann::json p_data)
{
	_id = p_data["id"].get<string>();
}


BaseLayer::~BaseLayer()
{
	if (_input_group != nullptr) delete _input_group;
	_input_group = nullptr;
	if (_output_group != nullptr) delete _output_group;
	_output_group = nullptr;
}

