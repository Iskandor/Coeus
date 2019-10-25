#include "GAE.h"

using namespace Coeus;

GAE::GAE(NeuralNetwork* p_network, GRADIENT_RULE p_grad_rule, float p_alpha, float p_gamma, float p_lambda) :
	_gamma(p_gamma),
	_lambda(p_lambda),
	_network(p_network)
{
	_value_estimator = new TD(p_network, p_grad_rule, p_alpha, p_gamma);
}


GAE::~GAE()
{
	delete _value_estimator;
}

void GAE::set_sample(vector<DQItem> &p_sample) {
	_sample_buffer = p_sample;
}

vector<float> GAE::get_advantages()
{
	vector<float> advantages;
	float advantage = 0;
	float gl = _gamma * _lambda;
	float Vs0 = 0;
	float Vs1 = 0;

	for (int l = 0; l < _sample_buffer.size(); l++) {
		_network->activate(&_sample_buffer[l].s0);
		Vs0 = _network->get_output()->at(0);
		_network->activate(&_sample_buffer[l].s1);
		Vs1 = _network->get_output()->at(0);

		advantage += pow(gl, l) *  (_sample_buffer[l].r + Vs1 - Vs0);
		advantages.push_back(advantage);
	}

	return advantages;
}


void GAE::train() {
	for (int l = 0; l < _sample_buffer.size(); l++) {
		_value_estimator->train(&_sample_buffer[l].s0, &_sample_buffer[l].s1, _sample_buffer[l].r, _sample_buffer[l].final);
	}
}