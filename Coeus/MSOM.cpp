#include <iostream>
#include "MSOM.h"
#include "IOUtils.h"
#include <future>
#include "ThreadPool.h"

using namespace Coeus;

MSOM::MSOM(int p_input_dim, int p_dim_x, int p_dim_y, NeuralGroup::ACTIVATION p_activation, double p_alpha, double p_beta) : SOM(p_input_dim, p_dim_x, p_dim_y, p_activation) {
	_context_group = new NeuralGroup(p_input_dim, NeuralGroup::LINEAR, false);
	_context_lattice = new Connection(_context_group->getDim(), _output_group->getDim(), _context_group->getId(), _output_group->getId());
	_context_lattice->init(Connection::UNIFORM, 0.1);
	_alpha = p_alpha;
	_beta = p_beta;
	_type = TYPE::MSOM;
}

MSOM::MSOM(nlohmann::json p_data) : SOM(p_data) {
	_type = TYPE::MSOM;
	_alpha = p_data["alpha"].get<double>();
	_beta = p_data["beta"].get<double>();

	_context_group = IOUtils::read_neural_group(p_data["groups"]["context"]);
	_context_lattice = IOUtils::read_connection(p_data["connections"]["context_lattice"]);
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
	double dc = 0;

	for (int i = 0; i < dim; i++) {
		if (_input_mask == nullptr || _input_mask[i] == 1) {
			dx += pow(xt->at(i) - xi->at(p_index, i), 2);
			dc += pow(ct->at(i) - ci->at(p_index, i), 2);
		}
	}

	return (1 - _alpha) * dx + _alpha * dc;;
}

MSOM * Coeus::MSOM::clone()
{
	MSOM* result = new MSOM(_input_group->getDim(), _dim_x, _dim_y, _output_group->getActivationFunction(), _alpha, _beta);

	result->_input_lattice = new Connection(*_input_lattice);
	result->_context_lattice = new Connection(*_context_lattice);

	return result;
}

void MSOM::calc_distance() {
	for (int l = 0; l < _dim_x * _dim_y; l++) {
		_dist.set(l, calc_distance(l));
	}
}

void MSOM::reset_context() const {
	_context_group->getOutput()->fill(0);
}

/*
void MSOM::calc_distance() {
	int i = 0;
	vector< future<double> > results;

	ThreadPool pool(_dim_x * _dim_y);

	for (int l = 0; l < _dim_x * _dim_y; l++) {
		results.emplace_back(
			pool.enqueue(
				[this] (int p_index) 
				{
					const int dim = _input_group->getDim();
					Tensor* xi = _input_lattice->get_weights();
					Tensor* ci = _context_lattice->get_weights();
					Tensor* xt = _input_group->getOutput();
					Tensor* ct = _context_group->getOutput();

					double dx = 0;
					double dc = 0;

					for (int i = 0; i < dim; i++) {
						dx += pow(xt->at(i) - xi->at(p_index, i), 2);
						dc += pow(ct->at(i) - ci->at(p_index, i), 2);
					}

					return (1 - _alpha) * dx + _alpha * dc;
				},
				l
			)
		);
	}

	for (auto && result : results) {
		_dist.set(i, result.get());
		i++;
	}
}
*/

void MSOM::update_context() const {
	Tensor* ct = _context_group->getOutput();

	Tensor* wIt = _input_lattice->get_weights();
	Tensor* cIt = _context_lattice->get_weights();

	for (int i = 0; i < _context_group->getDim(); i++) {
		ct->set(i, (1 - _beta) * wIt->at(_winner, i) + _beta * cIt->at(_winner, i));
	}
}
