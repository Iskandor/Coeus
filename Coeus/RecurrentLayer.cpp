#include "RecurrentLayer.h"
#include "RecurrentLayerGradient.h"
#include "IDGen.h"

using namespace Coeus;

RecurrentLayer::RecurrentLayer(const string p_id, const int p_dim, const NeuralGroup::ACTIVATION p_activation) : BaseLayer(p_id)
{
	_input_group = add_group(new NeuralGroup(p_dim, p_activation, true));
	_output_group = _input_group;
	_context_group = add_group(new NeuralGroup(p_dim, NeuralGroup::ACTIVATION::LINEAR, false));

	_rec_connection = add_connection(new Connection(_context_group->get_dim(), _output_group->get_dim(), _context_group->get_id(), _output_group->get_id()));

	_type = RECURRENT;

	_gradient_component = new RecurrentLayerGradient(this);
}

RecurrentLayer::RecurrentLayer(RecurrentLayer& p_copy) : BaseLayer(IDGen::instance().next()) {
	_input_group = add_group(new NeuralGroup(*p_copy._input_group));
	_output_group = _input_group;
	_context_group = add_group(new NeuralGroup(*p_copy._context_group));

	_rec_connection = add_connection(new Connection(_context_group->get_dim(), _output_group->get_dim(), _context_group->get_id(), _output_group->get_id()));
	_rec_connection->override(p_copy._rec_connection);

	_type = RECURRENT;

	_gradient_component = new RecurrentLayerGradient(this);
}

RecurrentLayer::~RecurrentLayer()
{
	delete _input_group;
	delete _context_group;
	delete _rec_connection;
	delete _gradient_component;
}

void RecurrentLayer::integrate(Tensor* p_input, Tensor* p_weights) {
	_output_group->integrate(p_input, p_weights);
}

void RecurrentLayer::activate(Tensor * p_input)
{
	_context_group->set_output(_output_group->get_output());
	_output_group->integrate(_context_group->get_output(), _rec_connection->get_weights());
	_output_group->activate();
}

void RecurrentLayer::override(BaseLayer * p_source)
{
	RecurrentLayer *source = dynamic_cast<RecurrentLayer*>(p_source);
	_input_group->get_bias()->override(source->_input_group->get_bias());
	_rec_connection->override(source->_rec_connection);
}