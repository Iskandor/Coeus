#include "NetworkGradient.h"
#include "IGradientComponent.h"

using namespace Coeus;

NetworkGradient::NetworkGradient(NeuralNetwork* p_network, ICostFunction* p_cost_function)
{
	_network = p_network;
	_cost_function = p_cost_function;

	for (auto it = _network->_backward_graph.begin(); it != _network->_backward_graph.end(); ++it) {
		if ((*it)->gradient_component() != nullptr) {
			(*it)->gradient_component()->init();
		}
	}
}


NetworkGradient::~NetworkGradient()
{
}

void NetworkGradient::calc_gradient(Tensor* p_target) {

	Tensor error = _cost_function->cost_deriv(_network->get_output(), p_target);

	for(auto it = _network->_backward_graph.begin(); it != _network->_backward_graph.end(); ++it) {
		if ((*it)->gradient_component() != nullptr) {
			(*it)->gradient_component()->calc_deriv();
		}
	}

	BaseLayer* output_layer = _network->_layers[_network->_output_layer];

	Tensor delta = Tensor::apply(error, *output_layer->gradient_component()->get_output_deriv(), Tensor::ew_dot);

	output_layer->gradient_component()->set_delta(&delta);

	BaseLayer* prev_layer = output_layer;

	for (auto it = ++_network->_backward_graph.begin(); it != _network->_backward_graph.end(); ++it) {
		if ((*it)->gradient_component() != nullptr) {
			(*it)->gradient_component()->calc_delta(_network->get_connection((*it)->id(), prev_layer->id())->get_weights(), prev_layer->gradient_component()->get_input_delta());
			prev_layer = *it;
		}		
	}

	for (auto it = _network->_backward_graph.begin(); it != _network->_backward_graph.end(); ++it) {
		if ((*it)->gradient_component() != nullptr) {
			(*it)->gradient_component()->calc_gradient(_w_gradient, _b_gradient);
		}
	}

	for (auto it = _network->_connections.begin(); it != _network->_connections.end(); ++it) {
		_w_gradient[(*it).first] = *_network->_layers[it->second->get_in_id()]->get_output() * *_network->_layers[it->second->get_out_id()]->gradient_component()->get_input_delta();
	}
}

void NetworkGradient::update(map<string, Tensor> &p_update) const {
	for (auto it = _network->_connections.begin(); it != _network->_connections.end(); ++it) {
		it->second->update_weights(p_update[it->first]);
	}

	for (auto it = _network->_backward_graph.begin(); it != _network->_backward_graph.end(); ++it) {
		if ((*it)->gradient_component() != nullptr) {
			(*it)->gradient_component()->update(p_update);
		}
	}
}
