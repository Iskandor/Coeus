#include "NetworkGradient.h"
#include "IGradientComponent.h"
#include "CoreLayerGradient.h"
#include "RecurrentLayerGradient.h"
#include "LSTMLayerGradient.h"

using namespace Coeus;

NetworkGradient::NetworkGradient(NeuralNetwork* p_network)
{
	_network = p_network;

	for (auto it = _network->_backward_graph.begin(); it != _network->_backward_graph.end(); ++it) {
		IGradientComponent* component = create_component(*it);
		if (component != nullptr) {
			component->init();
			_gradient_component[(*it)->get_id()] = component;
		}
	}

	_gradient = get_empty_params();
}


NetworkGradient::~NetworkGradient()
{
}

void NetworkGradient::calc_gradient(Tensor* p_value) {

	for(auto it = _network->_backward_graph.begin(); it != _network->_backward_graph.end(); ++it) {
		if (_gradient_component[(*it)->get_id()] != nullptr) {
			_gradient_component[(*it)->get_id()]->calc_deriv();
		}
	}

	BaseLayer* output_layer = _network->_layers[_network->_output_layer];

	Tensor delta;

	if (p_value == nullptr)
	{
		delta = *_gradient_component[output_layer->get_id()]->get_output_deriv() * Tensor::Ones({_network->get_output()->size()});
	}
	else
	{
		delta = *_gradient_component[output_layer->get_id()]->get_output_deriv() * *p_value;
	}
	

	_gradient_component[output_layer->get_id()]->set_delta(&delta);

	BaseLayer* prev_layer = output_layer;

	for (auto it = ++_network->_backward_graph.begin(); it != _network->_backward_graph.end(); ++it) {
		if (_gradient_component[(*it)->get_id()] != nullptr) {
			_gradient_component[(*it)->get_id()]->calc_delta(_network->get_connection((*it)->get_id(), prev_layer->get_id())->get_weights(), _gradient_component[prev_layer->get_id()]->get_input_delta());
			prev_layer = *it;
		}		
	}

	for (auto it = _network->_backward_graph.begin(); it != _network->_backward_graph.end(); ++it) {
		if (_gradient_component[(*it)->get_id()] != nullptr) {
			_gradient_component[(*it)->get_id()]->calc_gradient(_gradient);
		}
	}
}

void NetworkGradient::check_gradient(Tensor* p_input, Tensor* p_target, ICostFunction* p_loss) {
	const double epsilon = 1e-4;

	for (auto it = _network->_connections.begin(); it != _network->_connections.end(); ++it) {
		if (it->second->is_trainable())
		{
			cout << it->second->get_id() << endl;
			for(int i = 0; i < it->second->get_weights()->size(); i++) {
				const double w = it->second->get_weights()->at(i);
				it->second->get_weights()->set(i, w + epsilon);
				const double Je_plus = check_estimate(p_input, p_target, p_loss);
				it->second->get_weights()->set(i, w - epsilon);
				const double Je_minus = check_estimate(p_input, p_target, p_loss);

				it->second->get_weights()->set(i, w);

				const double de = (Je_plus - Je_minus) / (2 * epsilon);

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
					const double w = c->second->get_weights()->at(i);
					c->second->get_weights()->set(i, w + epsilon);
					const double Je_plus = check_estimate(p_input, p_target, p_loss);
					c->second->get_weights()->set(i, w - epsilon);
					const double Je_minus = check_estimate(p_input, p_target, p_loss);

					c->second->get_weights()->set(i, w);

					const double de = (Je_plus - Je_minus) / (2 * epsilon);

					cout << i << " " << _gradient[c->second->get_id()].at(i) - de << endl;
				}
			}

			for (auto g = (*it)->get_groups()->begin(); g != (*it)->get_groups()->end(); ++g) {				
				cout << g->second->get_id() << endl;

				for (int i = 0; i < g->second->get_bias()->size(); i++) {
					const double b = g->second->get_bias()->at(i);
					g->second->get_bias()->set(i, b + epsilon);
					const double Je_plus = check_estimate(p_input, p_target, p_loss);
					g->second->get_bias()->set(i, b - epsilon);
					const double Je_minus = check_estimate(p_input, p_target, p_loss);

					g->second->get_bias()->set(i, b);

					const double de = (Je_plus - Je_minus) / (2 * epsilon);

					cout << i << " " << _gradient[g->second->get_id()].at(i) - de << endl;
				}
			}
		}
	}
}

void NetworkGradient::reset()
{
	for (auto it = _network->_backward_graph.begin(); it != _network->_backward_graph.end(); ++it) {
		if (_gradient_component[(*it)->get_id()] != nullptr) {
			_gradient_component[(*it)->get_id()]->reset();
		}
	}
}

map<string, Tensor> NetworkGradient::get_empty_params() const
{
	map<string, Tensor> result;

	for(auto it = _network->_params.begin(); it != _network->_params.end(); ++it)
	{
		result[it->first] = Tensor(it->second->rank(), it->second->shape(), Tensor::ZERO);
	}

	return result;
}

IGradientComponent* NetworkGradient::create_component(BaseLayer* p_layer) const {
	IGradientComponent* component = nullptr;

	switch(p_layer->get_type()) {
		case BaseLayer::SOM: break;
		case BaseLayer::MSOM: break;
		case BaseLayer::INPUT: break;
		case BaseLayer::CORE: 
			component = new CoreLayerGradient(p_layer, _network);
		break;
		case BaseLayer::RECURRENT: 
			component = new RecurrentLayerGradient(p_layer, _network);
		break;
		case BaseLayer::LSTM: 
			component = new LSTMLayerGradient(p_layer, _network);
		break;
		case BaseLayer::LSOM: break;
		default: ;
	}

	return component;
}

double NetworkGradient::check_estimate(Tensor* p_input, Tensor* p_target, ICostFunction* p_loss) const {
	_network->activate(p_input);
	return p_loss->cost(_network->get_output(), p_target);
}

void NetworkGradient::activate(Tensor* p_input)
{
	// single input
	if (p_input->rank() == 1)
	{
		_network->_layers[_network->_input_layer[0]]->activate(p_input);

		_network->activate();
		calc_deriv_estimate();
	}

	// sequence
	if (p_input->rank() == 2)
	{
		Tensor input = Tensor::Zero({ p_input->shape(1) });

		reset();
		for (int i = 0; i < p_input->shape(0); i++)
		{
			p_input->get_row(input, i);
			_network->_layers[_network->_input_layer[0]]->activate(&input);

			_network->activate();
			calc_deriv_estimate();
		}
	}
}

void NetworkGradient::calc_deriv_estimate()
{
	for (auto it = _network->_backward_graph.begin(); it != _network->_backward_graph.end(); ++it) {
		if (_gradient_component[(*it)->get_id()] != nullptr) {
			_gradient_component[(*it)->get_id()]->calc_deriv_estimate();
		}
	}
}
