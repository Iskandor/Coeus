#include "InputLayer.h"
#include "IDGen.h"

using namespace Coeus;

InputLayer::InputLayer(const string p_id, const int p_input_dim) : BaseLayer(p_id)
{
	_input_group = add_group(new NeuralGroup(p_input_dim, NeuralGroup::ACTIVATION::LINEAR, false));
	_output_group = _input_group;
	_type = INPUT;
}

InputLayer::InputLayer(InputLayer& p_copy) : BaseLayer(IDGen::instance().next()) {
	_input_group = add_group(new NeuralGroup(*p_copy._input_group));
	_output_group = _input_group;
	_type = INPUT;
}

InputLayer::~InputLayer()
{
	delete _input_group;
}

void InputLayer::integrate(Tensor* p_input, Tensor* p_weights) {
}

void InputLayer::activate(Tensor * p_input)
{
	if (p_input != nullptr) {
		_output_group->set_output(p_input);
	}	
}

void InputLayer::override(BaseLayer * p_source)
{
}