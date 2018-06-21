#include "CoreLayer.h"
#include "CoreLayerGradient.h"

using namespace Coeus;

CoreLayer::CoreLayer(const string p_id, const int p_dim, const NeuralGroup::ACTIVATION p_activation) : BaseLayer(p_id)
{
	_output_group = new NeuralGroup(p_dim, p_activation, true);
	_input_group = _output_group;

	_type = CORE;
	_gradient_component = new CoreLayerGradient(this);
}

CoreLayer::~CoreLayer()
{
	delete _output_group;
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

void CoreLayer::post_connection(BaseLayer* p_input)
{
}
