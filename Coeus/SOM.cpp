#include "SOM.h"
#include "ActivationFunctions.h"
#include <chrono>
#include "Connection.h"
#include "IOUtils.h"

using namespace Coeus;

SOM::SOM(const int p_input_dim, const int p_dim_x, const int p_dim_y, const NeuralGroup::ACTIVATION p_activation)
{
	_type = TYPE::SOM;
	_dim_x = p_dim_x;
	_dim_y = p_dim_y;

	_input_group = new NeuralGroup(p_input_dim, NeuralGroup::ACTIVATION::LINEAR, false);
	_output_group = new NeuralGroup(p_dim_x * p_dim_y, p_activation, false);

	_input_lattice = new Connection(_input_group->getDim(), _output_group->getDim(), _input_group->getId(), _output_group->getId());
	_input_lattice->init(Connection::UNIFORM, 0.1);

	_dist = Tensor::Zero({ _dim_x * _dim_y });
	_input_mask = nullptr;
}

SOM::SOM(nlohmann::json p_data) {
	_type = TYPE::SOM;

	_dim_x = p_data["dim_x"].get<int>();
	_dim_y = p_data["dim_y"].get<int>();

	_input_group = IOUtils::read_neural_group(p_data["groups"]["input"]);
	_output_group = IOUtils::read_neural_group(p_data["groups"]["lattice"]);
	_input_lattice = IOUtils::read_connection(p_data["connections"]["input_lattice"]);
}


SOM::~SOM()
{
	if (_input_lattice != nullptr) delete _input_lattice;
	_input_lattice = nullptr;
}

void SOM::activate(Tensor* p_input) {
	find_winner(p_input);

	calc_distance();

	switch (_output_group->getActivationFunction()) {
		case NeuralGroup::LINEAR:
			_dist = _dist.apply(ActivationFunctions::linear);
			break;
		case NeuralGroup::EXPONENTIAL:
			_dist = _dist.apply(ActivationFunctions::exponential);
			break;
		case NeuralGroup::KEXPONENTIAL:
			_dist = _dist.apply(ActivationFunctions::kexponential);
			break;
		case NeuralGroup::GAUSS:
			_dist = _dist.apply(ActivationFunctions::gauss);
			break;
		default:
			break;
	}

	_output_group->setOutput(_dist);
}

double SOM::calc_distance(const int p_index) {
	const int dim = _input_group->getDim();
	double s = 0;

	for (int i = 0; i < dim; i++) {
		if (_input_mask == nullptr || _input_mask[i] == 1) {
			s += pow(_input_group->getOutput()->at(i) - _input_lattice->get_weights()->at(p_index, i), 2);
		}
	}

	return sqrt(s);
}

void SOM::calc_distance() {
	for(int l = 0; l < _dim_x * _dim_y; l++) {
		_dist.set(l, calc_distance(l));
	}
}

int SOM::find_winner(Tensor* p_input) {
	double winner_dist = INFINITY;
	_winner = 0;

	_input_group->setOutput(*p_input);

	for (int i = 0; i < _output_group->getDim(); i++) {
		const double neuron_dist = calc_distance(i);
		if (winner_dist > neuron_dist) {
			_winner = i;
			winner_dist = neuron_dist;
		}
	}

	/*
	calc_distance();
	_winner = _dist.max_index();
	*/

	return _winner;
}

void SOM::get_position(const int p_index, int& p_x, int& p_y) const {
	p_x = p_index % _dim_x;
	p_y = p_index / _dim_x;

}
