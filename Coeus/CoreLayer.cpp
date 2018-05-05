#include "CoreLayer.h"
#include "CoreLayerGradient.h"

using namespace Coeus;

CoreLayer::CoreLayer(const string p_id, const int p_dim, const NeuralGroup::ACTIVATION p_activation, const bool p_bias) : BaseLayer(p_id)
{
	_input_group = new NeuralGroup(p_dim, p_activation, p_bias);
	_output_group = _input_group;

	_type = CORE;
	_gradient_component = new CoreLayerGradient(this);
}

CoreLayer::~CoreLayer()
{
	delete _input_group;
	delete _gradient_component;
}

void CoreLayer::integrate(Tensor* p_input, Tensor* p_weights) {
	_output_group->integrate(p_input, p_weights);
}

void CoreLayer::activate(Tensor * p_input)
{	
	_output_group->activate();
}

void CoreLayer::override_params(BaseLayer * p_source)
{
}
