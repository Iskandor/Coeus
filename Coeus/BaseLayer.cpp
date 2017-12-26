#include "BaseLayer.h"

using namespace Coeus;

BaseLayer::BaseLayer(): _input_group(nullptr), _output_group(nullptr) 
{
}


BaseLayer::~BaseLayer()
{
	if (_input_group != nullptr) delete _input_group;
	_input_group = nullptr;
	if (_output_group != nullptr) delete _output_group;
	_output_group = nullptr;
}

