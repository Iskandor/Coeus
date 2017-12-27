#include "MSOM.h"

using namespace Coeus;

MSOM::MSOM(int p_input_dim, int p_dim_x, int p_dim_y, NeuralGroup::ACTIVATION p_activation, double p_alpha, double p_beta) : SOM(p_input_dim, p_dim_x, p_dim_y, p_activation) {
	_context_group = new NeuralGroup(p_input_dim, NeuralGroup::LINEAR, false);
	_context_lattice = new Connection(_context_group->getDim(), _output_group->getDim(), _context_group->getId(), _output_group->getId());
	_alpha = p_alpha;
	_beta = p_beta;
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
	const int dim = _input_group->getDim();

	Tensor* xi = _input_lattice->get_weights();
	Tensor* ci = _context_lattice->get_weights();
	Tensor* xt = _input_group->getOutput();
	Tensor* ct = _context_group->getOutput();

	double dx = 0;

	for (int i = 0; i < dim; i++) {
		dx += pow(xt->at(i) - xi->at(p_index, i), 2);
	}

	double dc = 0;

	for (int i = 0; i < dim; i++) {
		dc += pow(ct->at(i) - ci->at(p_index, i), 2);
	}

	const double dt = (1 - _alpha) * dx + _alpha * dc;
	return dt;
}

void MSOM::reset_context() const {
	_context_group->getOutput()->fill(0);
}

Tensor* MSOM::calc_distance() {
	const int dim = _input_group->getDim();

	Tensor* xi = _input_lattice->get_weights();
	Tensor* ci = _context_lattice->get_weights();
	Tensor* xt = _input_group->getOutput();
	Tensor* ct = _context_group->getOutput();

	double* arr = new double[_dim_x * _dim_y];

	for (int l = 0; l < _dim_x * _dim_y; l++) {

		double dx = 0;

		for (int i = 0; i < dim; i++) {
			dx += pow(xt->at(i) - xi->at(l, i), 2);
		}

		double dc = 0;

		for (int i = 0; i < dim; i++) {
			dc += pow(ct->at(i) - ci->at(l, i), 2);
		}

		arr[l] = (1 - _alpha) * dx + _alpha * dc;
	}

	return new Tensor({ _dim_x, _dim_y }, arr);
}

void MSOM::update_context() const {
	Tensor* ct = _context_group->getOutput();

	Tensor* wIt = _input_lattice->get_weights();
	Tensor* cIt = _context_lattice->get_weights();

	for (int i = 0; i < _context_group->getDim(); i++) {
		ct->set(i, (1 - _beta) * wIt->at(_winner, i) + _beta * cIt->at(_winner, i));
	}
}
