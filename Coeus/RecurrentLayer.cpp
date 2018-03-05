#include "RecurrentLayer.h"

using namespace Coeus;

RecurrentLayer::RecurrentLayer(string p_id, int p_dim, NeuralGroup::ACTIVATION p_activation, NeuralGroup * p_parent) : BaseLayer(p_id)
{
	_input_group = p_parent;
	_output_group = new NeuralGroup(p_dim, p_activation, true);
	_context_group = new NeuralGroup(p_dim, NeuralGroup::ACTIVATION::LINEAR, true);

	_in_connection = new Connection(_input_group->getDim(), _output_group->getDim(), _input_group->getId(), _output_group->getId());
	_rec_connection = new Connection(_context_group->getDim(), _output_group->getDim(), _context_group->getId(), _output_group->getId());

	_type = BaseLayer::CORE;
}

RecurrentLayer::~RecurrentLayer()
{
	delete _output_group;
	delete _context_group;
	delete _in_connection;
	delete _rec_connection;
}

void RecurrentLayer::activate(Tensor * p_input)
{
	_context_group->setOutput(_output_group->getOutput());
	_output_group->integrate(p_input, _in_connection->get_weights());
	_output_group->integrate(_context_group->getOutput(), _rec_connection->get_weights());
	_output_group->activate();
}

void RecurrentLayer::override_params(BaseLayer * p_source)
{
}
