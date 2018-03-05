#include "InputLayer.h"

using namespace Coeus;

InputLayer::InputLayer(string p_id, const int p_input_dim) : BaseLayer(p_id)
{
	_output_group = new NeuralGroup(p_input_dim, NeuralGroup::ACTIVATION::LINEAR, false);
	_type = BaseLayer::INPUT;
}

InputLayer::~InputLayer()
{
	delete _output_group;
}

void InputLayer::activate(Tensor * p_input)
{
	_output_group->setOutput(p_input);
}

void InputLayer::override_params(BaseLayer * p_source)
{
}
