#include "GAE.h"

using namespace Coeus;

GAE::GAE(NeuralNetwork* p_network, float p_gamma, float p_lambda) :
	_gamma(p_gamma),
	_lambda(p_lambda),
	_network(p_network)
{
	_network_gradient = new NetworkGradient(p_network);
}


GAE::~GAE()
{
	delete _network_gradient;
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

map<string, Tensor>& GAE::get_gradient(Tensor* p_state0, float p_advantage) const
{
	_network->activate(p_state0);
	const float Vs0 = _network->get_output()->at(0);

	Tensor loss({ 1 }, Tensor::VALUE, Vs0 - p_advantage);

	_network_gradient->calc_gradient(&loss);

	return _network_gradient->get_gradient();
}
