#include "CoreLayer.h"

using namespace Coeus;

CoreLayer::CoreLayer(string p_id, int p_dim, NeuralGroup::ACTIVATION p_activation) : BaseLayer(p_id)
{
	_input_group = new NeuralGroup(p_dim, p_activation, true);
	_output_group = _input_group;

	_type = BaseLayer::CORE;
}

CoreLayer::~CoreLayer()
{
	delete _input_group;
}

void CoreLayer::activate(Tensor * p_input, Tensor* p_weights)
{
	_output_group->integrate(p_input, p_weights);
	_output_group->activate();
}

void Coeus::CoreLayer::override_params(BaseLayer * p_source)
{
}
