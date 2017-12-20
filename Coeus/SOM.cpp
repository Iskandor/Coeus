#include "SOM.h"
#include "ActivationFunctions.h"

using namespace Coeus;

SOM::SOM(const int p_input_dim, const int p_dim_x, const int p_dim_y, const NeuralGroup::ACTIVATION p_activation)
{
	NeuralGroup* input = add_group(p_input_dim, NeuralGroup::ACTIVATION::LINEAR, false);
	NeuralGroup* lattice = add_group(p_dim_x * p_dim_y, p_activation, false);

	add_connection(input, lattice, Connection::GLOROT_UNIFORM, 0.1);

	_lattice = _outputGroup;
}


SOM::~SOM()
{
}

void SOM::activate(Tensor* p_input) {
	find_winner(p_input);

	Tensor* output = calc_distance();

	switch (_groups[_lattice]->getActivationFunction()) {
		case NeuralGroup::LINEAR:
			output->apply(ActivationFunctions::linear);
			break;
		case NeuralGroup::EXPONENTIAL:
			output->apply(ActivationFunctions::exponential);
			break;
		case NeuralGroup::KEXPONENTIAL:
			output->apply(ActivationFunctions::kexponential);
			break;
		case NeuralGroup::GAUSS:
			output->apply(ActivationFunctions::gauss);
			break;
		default:
			break;
	}

	_groups[_lattice]->setOutput(output);
	delete output;
}

double SOM::calc_distance(const int p_index) {
	Tensor* input = _groups[_inputGroup]->getOutput();
	Tensor* weights = get_connection(_inputGroup, _lattice)->get_weights();

	int dim = _groups[_inputGroup]->getDim();
	double s = 0;

	for (int i = 0; i < dim; i++) {
		s += pow(input->at(i) - weights->at(p_index, i), 2);
	}

	return sqrt(s);
}

Tensor* SOM::calc_distance() {
	Tensor* input = _groups[_inputGroup]->getOutput();
	Tensor* weights = get_connection(_inputGroup, _lattice)->get_weights();

	const int l_dim = _groups[_lattice]->getDim();
	const int i_dim = _groups[_inputGroup]->getDim();
	double* arr = new double[l_dim];

	for(int l = 0; l < l_dim; l++) {
		double s = 0;
		for (int i = 0; i < i_dim; i++) {
			s += pow(input->at(i) - weights->at(l, i), 2);
		}
		arr[l] = sqrt(s);
	}

	return new Tensor({ _dimX, _dimY}, arr);
}

int SOM::find_winner(Tensor* p_input) {
	double winner_dist = INFINITY;
	double neuron_dist = 0;
	_winner = 0;

	_groups[_inputGroup]->setOutput(p_input);

	for (int i = 0; i < _groups[_lattice]->getDim(); i++) {
		neuron_dist = calc_distance(i);
		if (winner_dist > neuron_dist) {
			_winner = i;
			winner_dist = neuron_dist;
		}
	}

	return _winner;
}

void SOM::get_position(const int p_index, int& p_x, int& p_y) const {
	p_x = p_index % _dimX;
	p_y = p_index / _dimX;

}
