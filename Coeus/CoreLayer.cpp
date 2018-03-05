#include "CoreLayer.h"

using namespace Coeus;

CoreLayer::CoreLayer(string p_id, int p_dim, NeuralGroup::ACTIVATION p_activation, NeuralGroup* p_parent) : BaseLayer(p_id)
{
	_input_group = p_parent;
	_output_group = new NeuralGroup(p_dim, p_activation, true);

	_in_connection = new Connection(_input_group->getDim(), _output_group->getDim(), _input_group->getId(), _output_group->getId());
	_type = BaseLayer::CORE;
}

CoreLayer::~CoreLayer()
{
	delete _output_group;
	delete _in_connection;
}

void CoreLayer::activate(Tensor * p_input)
{
	_output_group->integrate(p_input, _in_connection->get_weights());
	_output_group->activate();
}

void CoreLayer::override_params(BaseLayer * p_source)
{
	_in_connection->set_weights(static_cast<CoreLayer*>(p_source)->_in_connection->get_weights());
}
