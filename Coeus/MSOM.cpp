#include <iostream>
#include "MSOM.h"
#include "IOUtils.h"

using namespace Coeus;

MSOM::MSOM(string p_id, int p_input_dim, int p_dim_x, int p_dim_y, ACTIVATION p_activation, double p_alpha, double p_beta) : SOM(p_id, p_input_dim, p_dim_x, p_dim_y, p_activation) {
	_context_group = new SimpleCellGroup(p_input_dim, LINEAR, false);
	_context_lattice = new Connection(_context_group->get_dim(), _lattice_group->get_dim(), _context_group->get_id(), _lattice_group->get_id());
	_context_lattice->init(Connection::UNIFORM, true, 0.01);
	_alpha = p_alpha;
	_beta = p_beta;
	_type = TYPE::MSOM;
}

MSOM::MSOM(nlohmann::json p_data) : SOM(p_data) {
	_type = TYPE::MSOM;
	_alpha = p_data["alpha"].get<double>();
	_beta = p_data["beta"].get<double>();

	_context_group = new SimpleCellGroup(p_data["groups"]["context"]);
	_context_lattice = new Connection(p_data["connections"]["context_lattice"]);
}

MSOM::~MSOM()
{
	if (_context_group != nullptr) delete _context_group;
	_context_group = nullptr;
	if (_context_lattice != nullptr) delete _context_lattice;
	_context_lattice = nullptr;
}

void MSOM::activate(Tensor* p_input) {
	SOM::activate(p_input);
	update_context();
}

double MSOM::calc_distance(const int p_index) {
	const int dim = _input_group->get_dim();

	Tensor* xi = _afferent->get_weights();
	Tensor* ci = _context_lattice->get_weights();
	Tensor* xt = _input_group->get_output();
	Tensor* ct = _context_group->get_output();

	double dx = 0;
	double dc = 0;

	for (int i = 0; i < dim; i++) {
		if (_input_mask == nullptr || _input_mask[i] == 1) {
			dx += pow(xt->at(i) - xi->at(p_index, i), 2);
			dc += pow(ct->at(i) - ci->at(p_index, i), 2);
		}
	}

	return (1 - _alpha) * dx + _alpha * dc;
}

double MSOM::calc_distance(const int p_neuron1, const int p_neuron2)
{
	const int dim = _input_group->get_dim();

	Tensor* xi = _afferent->get_weights();
	Tensor* ci = _context_lattice->get_weights();

	double dx = 0;
	double dc = 0;

	for (int i = 0; i < dim; i++) {
		if (_input_mask == nullptr || _input_mask[i] == 1) {
			dx += pow(xi->at(p_neuron1, i) - xi->at(p_neuron2, i), 2);
			dc += pow(ci->at(p_neuron1, i) - ci->at(p_neuron2, i), 2);
		}
	}

	return (1 - _alpha) * dx + _alpha * dc;
}

MSOM * MSOM::clone() const {
	/*
	MSOM* result = dynamic_cast<MSOM*>(IOUtils::load_layer(IOUtils::save_layer((BaseLayer*)this)));

	return result;
	*/
	return nullptr;
}

void MSOM::override(BaseLayer * p_source)
{
	SOM::override(p_source);

	MSOM* msom = dynamic_cast<MSOM*>(p_source);

	_context_lattice->set_weights(msom->get_context_lattice()->get_weights());
}

void MSOM::reset_context() const {
	_context_group->get_output()->fill(0);
}

void MSOM::update_context() const {
	Tensor* ct = _context_group->get_output();

	Tensor* wIt = _afferent->get_weights();
	Tensor* cIt = _context_lattice->get_weights();

	for (int i = 0; i < _context_group->get_dim(); i++) {
		ct->set(i, (1 - _beta) * wIt->at(_winner, i) + _beta * cIt->at(_winner, i));
	}
}
