#include "SOM.h"
#include "ActivationFunctions.h"
#include <chrono>
#include "Connection.h"
#include "IOUtils.h"

using namespace Coeus;

SOM::SOM(string p_id, const int p_input_dim, const int p_dim_x, const int p_dim_y, const NeuralGroup::ACTIVATION p_activation) : BaseLayer(p_id)
{
	_type = TYPE::SOM;
	_dim_x = p_dim_x;
	_dim_y = p_dim_y;

	_input_group = new NeuralGroup(p_input_dim, NeuralGroup::ACTIVATION::LINEAR, false);
	_output_group = new NeuralGroup(p_dim_x * p_dim_y, p_activation, false);

	_afferent = new Connection(_input_group->get_dim(), _output_group->get_dim(), _input_group->get_id(), _output_group->get_id());
	_afferent->init(Connection::UNIFORM, 0.1);

	_dist = Tensor::Zero({ _dim_x * _dim_y });
	_p = Tensor::Zero({ _dim_x * _dim_y });
	_bias = Tensor::Zero({ _dim_x * _dim_y });
	_input_mask = nullptr;

	_conscience = 0;
}

SOM::SOM(nlohmann::json p_data) : BaseLayer(p_data) {
	_type = TYPE::SOM;

	_dim_x = p_data["dim_x"].get<int>();
	_dim_y = p_data["dim_y"].get<int>();

	_input_group = IOUtils::read_neural_group(p_data["groups"]["input"]);
	_output_group = IOUtils::read_neural_group(p_data["groups"]["lattice"]);
	_afferent = IOUtils::read_connection(p_data["connections"]["input_lattice"]);

	_dist = Tensor::Zero({ _dim_x * _dim_y });
	_p = Tensor::Zero({ _dim_x * _dim_y });
	_bias = Tensor::Zero({ _dim_x * _dim_y });
	_input_mask = nullptr;
}


SOM::~SOM()
{
	if (_afferent != nullptr) delete _afferent;
	_afferent = nullptr;
}

void SOM::integrate(Tensor* p_input, Tensor* p_weights) {
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

	_output_group->set_output(&_dist);
}

double SOM::calc_distance(const int p_index) {
	const int dim = _input_group->get_dim();
	double s = 0;

	for (int i = 0; i < dim; i++) {
		if (_input_mask == nullptr || _input_mask[i] == 1) {
			s += pow(_input_group->getOutput()->at(i) - _afferent->get_weights()->at(p_index, i), 2);
		}
	}

	return sqrt(s);
}

double SOM::calc_distance(const int p_neuron1, const int p_neuron2)
{
	const int dim = _input_group->get_dim();
	double s = 0;

	for (int i = 0; i < dim; i++) {
		s += pow(_afferent->get_weights()->at(p_neuron1, i) - _afferent->get_weights()->at(p_neuron2, i), 2);
	}

	return sqrt(s);
}

void SOM::set_conscience(const double p_val) {
	_conscience = p_val;
}

void SOM::init_conscience() const {
	_p.fill(0);
}

SOM * SOM::clone() const {
	SOM* result = new SOM(_id, _input_group->get_dim(), _dim_x, _dim_y, _output_group->getActivationFunction());

	result->_afferent = new Connection(*_afferent);
	result->_conscience = _conscience;

	return result;
}

void SOM::override_params(BaseLayer * p_source)
{
	SOM* som = static_cast<SOM*>(p_source);

	_afferent->set_weights(som->get_afferent()->get_weights());
	_conscience = som->_conscience;

	if (_conscience > 0) {
		_p.override(&som->_p);
		_bias.override(&som->_bias);
	}
}

void SOM::update_conscience(Tensor* p_input) {
	find_winner(p_input, false);

	for(int i = 0; i < _dim_y * _dim_x; i++) {
		const double p_new = _p.at(i) + B * (i == _winner ? 1 : 0 - _p.at(i));
		_p.set(i, p_new);
	}

	for (int i = 0; i < _dim_y * _dim_x; i++) {
		const double bias_new =  _conscience * (1/(_dim_y * _dim_x) - _p.at(i));
		_bias.set(i, bias_new);
	}
}

void SOM::calc_distance() {
	for(int l = 0; l < _dim_x * _dim_y; l++) {
		_dist.set(l, calc_distance(l));
	}
}

int SOM::find_winner(Tensor* p_input) {

	if (_conscience > 0) {
		find_winner(p_input, true);
	}
	else {
		find_winner(p_input, false);
	}

	return _winner;
}

void SOM::find_winner(Tensor* p_input, const bool p_conscience) {
	double winner_dist = INFINITY;
	_winner = 0;

	_input_group->set_output(p_input);

	for (int i = 0; i < _output_group->get_dim(); i++) {
		double neuron_dist = calc_distance(i);

		if (p_conscience) {
			neuron_dist -= _bias.at(i);
		}

		if (winner_dist > neuron_dist) {
			_winner = i;
			winner_dist = neuron_dist;
		}
	}
}

void SOM::get_position(const int p_index, int& p_x, int& p_y) const {
	p_x = p_index % _dim_x;
	p_y = p_index / _dim_x;
}

int SOM::get_position(const int p_x, const int p_y) const
{
	int pos = p_y * _dim_x + p_x;
	if (p_x < 0 || p_x >= _dim_x || p_y < 0 || p_y >= _dim_y) pos = -1;
	return pos;
}
