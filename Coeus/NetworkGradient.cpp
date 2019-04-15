#include "NetworkGradient.h"
#include "TensorOperator.h"
#include "NeuronOperator.h"

using namespace Coeus;

NetworkGradient::NetworkGradient(NeuralNetwork* p_network)
{
	_network = p_network;
	_gradient = _network->get_empty_params();
}


NetworkGradient::~NetworkGradient()
{
	for (auto& it : _delta)
	{
		delete it.second;
		it.second = nullptr;
	}

	for (auto& it : _derivative)
	{
		delete it.second;
		it.second = nullptr;
	}
}

void NetworkGradient::calc_gradient(Tensor* p_value) {

	BaseLayer* output_layer = _network->_layers[_network->_output_layer];

	if (p_value != nullptr)
	{
		if (p_value->rank() == 1)
		{
			_delta[_network->_output_layer] = NeuronOperator::init_auxiliary_parameter(_delta[_network->_output_layer], 1, output_layer->get_dim());
			TensorOperator::instance().vv_ewprod(p_value->arr(), _derivative[_network->_output_layer]->arr(), _delta[_network->_output_layer]->arr(), output_layer->get_dim());
		}
		if (p_value->rank() == 2)
		{
			_delta[_network->_output_layer] = NeuronOperator::init_auxiliary_parameter(_delta[_network->_output_layer], p_value->shape(0), output_layer->get_dim());
			TensorOperator::instance().vv_ewprod(p_value->arr(), _derivative[_network->_output_layer]->arr(), _delta[_network->_output_layer]->arr(), p_value->shape(0) * output_layer->get_dim());
		}
	}
	else
	{
		_delta[_network->_output_layer] = _derivative[_network->_output_layer];
	}

	for (auto& it : _network->_backward_graph)
	{
		it->calc_delta(_delta, _derivative);
		it->calc_gradient(_gradient, _delta, _derivative);
	}
}

void NetworkGradient::check_gradient(Tensor* p_input, Tensor* p_target, ICostFunction* p_loss) {
	/*
	const float epsilon = 1e-4f;

	for (auto it = _network->_connections.begin(); it != _network->_connections.end(); ++it) {
		if (it->second->is_trainable())
		{
			cout << it->second->get_id() << endl;
			for(int i = 0; i < it->second->get_weights()->size(); i++) {
				const float w = it->second->get_weights()->at(i);
				it->second->get_weights()->set(i, w + epsilon);
				const float Je_plus = check_estimate(p_input, p_target, p_loss);
				it->second->get_weights()->set(i, w - epsilon);
				const float Je_minus = check_estimate(p_input, p_target, p_loss);

				it->second->get_weights()->set(i, w);

				const float de = (Je_plus - Je_minus) / (2 * epsilon);

				cout << i << " " << _gradient[it->second->get_id()].at(i) - de << endl;
			}
			
		}
	}

	for (auto it = _network->_backward_graph.begin(); it != _network->_backward_graph.end(); ++it) {
		if (_gradient_component[(*it)->get_id()] != nullptr) {
			cout << (*it)->get_id() << endl;

			for (auto c = (*it)->get_connections()->begin(); c != (*it)->get_connections()->end(); ++c) {
				cout << c->second->get_id() << endl;
				for (int i = 0; i < c->second->get_weights()->size(); i++) {
					const float w = c->second->get_weights()->at(i);
					c->second->get_weights()->set(i, w + epsilon);
					const float Je_plus = check_estimate(p_input, p_target, p_loss);
					c->second->get_weights()->set(i, w - epsilon);
					const float Je_minus = check_estimate(p_input, p_target, p_loss);

					c->second->get_weights()->set(i, w);

					const float de = (Je_plus - Je_minus) / (2 * epsilon);

					cout << i << " " << _gradient[c->second->get_id()].at(i) - de << endl;
				}
			}

			for (auto g = (*it)->get_groups()->begin(); g != (*it)->get_groups()->end(); ++g) {				
				cout << g->second->get_id() << endl;

				for (int i = 0; i < g->second->get_bias()->size(); i++) {
					const float b = g->second->get_bias()->at(i);
					g->second->get_bias()->set(i, b + epsilon);
					const float Je_plus = check_estimate(p_input, p_target, p_loss);
					g->second->get_bias()->set(i, b - epsilon);
					const float Je_minus = check_estimate(p_input, p_target, p_loss);

					g->second->get_bias()->set(i, b);

					const float de = (Je_plus - Je_minus) / (2 * epsilon);

					cout << i << " " << _gradient[g->second->get_id()].at(i) - de << endl;
				}
			}
		}
	}
	*/
}

void NetworkGradient::reset()
{
	_network->reset();

	for (auto& it : _derivative)
	{
		it.second->fill(0);
	}
}

void NetworkGradient::calc_derivative()
{
	for (auto& it : _network->_backward_graph)
	{
		it->calc_derivative(_derivative);
	}
}

float NetworkGradient::check_estimate(Tensor* p_input, Tensor* p_target, ICostFunction* p_loss) const {
	_network->activate(p_input);
	return p_loss->cost(_network->get_output(), p_target);
}

void NetworkGradient::activate(Tensor* p_input)
{
	_network->_layers[_network->_input_layer[0]]->integrate(p_input);
	_network->activate();
	calc_derivative();
}

void NetworkGradient::activate(vector<Tensor*>* p_input)
{	
	reset();
	for (auto& it : *p_input)
	{
		_network->_layers[_network->_input_layer[0]]->integrate(it);
		_network->activate();
		calc_derivative();
	}
}