#include "RecurrentLayer.h"
#include "RecurrentLayerGradient.h"
#include "IDGen.h"

using namespace Coeus;

RecurrentLayer::RecurrentLayer(const string& p_id, const int p_dim, const ACTIVATION p_activation) : BaseLayer(p_id)
{
	_group = add_group(new SimpleCellGroup(p_dim, p_activation, true));
	_input_group = _output_group = _group;
	_context_group = add_group(new SimpleCellGroup(p_dim, LINEAR, false));

	_rec_connection = add_connection(new Connection(_context_group->get_dim(), _group->get_dim(), _context_group->get_id(), _group->get_id()));

	_type = RECURRENT;

	_gradient_component = new RecurrentLayerGradient(this);
}

RecurrentLayer::RecurrentLayer(RecurrentLayer& p_copy) : BaseLayer(IDGen::instance().next()) {
	_group = add_group<SimpleCellGroup>(p_copy._group->clone());
	_input_group = _output_group = _group;
	_context_group = add_group<SimpleCellGroup>(p_copy._context_group->clone());

	_rec_connection = add_connection(new Connection(_context_group->get_dim(), _group->get_dim(), _context_group->get_id(), _group->get_id()));
	_rec_connection->override(p_copy._rec_connection);

	_type = RECURRENT;

	_gradient_component = new RecurrentLayerGradient(this);
}

RecurrentLayer::~RecurrentLayer()
{
	delete _group;
	delete _context_group;
	delete _rec_connection;
	delete _gradient_component;
}

void RecurrentLayer::integrate(Tensor* p_input, Tensor* p_weights) {
	_group->integrate(p_input, p_weights);
}

void RecurrentLayer::activate(Tensor * p_input)
{
	_context_group->set_output(_group->get_output());
	_group->integrate(_context_group->get_output(), _rec_connection->get_weights());
	_group->activate();
}

void RecurrentLayer::override(BaseLayer * p_source)
{
	RecurrentLayer *source = dynamic_cast<RecurrentLayer*>(p_source);
	_group->get_bias()->override(source->_group->get_bias());
	_rec_connection->override(source->_rec_connection);
}