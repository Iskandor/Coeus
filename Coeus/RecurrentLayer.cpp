#include "RecurrentLayer.h"

using namespace Coeus;

RecurrentLayer::RecurrentLayer(string p_id, int p_dim, NeuralGroup::ACTIVATION p_activation) : BaseLayer(p_id)
{
	_input_group = new NeuralGroup(p_dim, p_activation, true);
	_output_group = _input_group;
	_context_group = new NeuralGroup(p_dim, NeuralGroup::ACTIVATION::LINEAR, true);

	_rec_connection = new Connection(_context_group->get_dim(), _output_group->get_dim(), _context_group->get_id(), _output_group->get_id());

	_type = BaseLayer::CORE;
}

RecurrentLayer::~RecurrentLayer()
{
	delete _input_group;
	delete _context_group;
	delete _rec_connection;
}

void RecurrentLayer::activate(Tensor * p_input, Tensor* p_weights)
{
	_context_group->setOutput(_output_group->getOutput());
	_output_group->integrate(p_input, p_weights);
	_output_group->integrate(_context_group->getOutput(), _rec_connection->get_weights());
	_output_group->activate();
}

void RecurrentLayer::override_params(BaseLayer * p_source)
{
}
