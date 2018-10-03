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

	Tensor delta = *output_layer->gradient_component()->get_output_deriv() * error;

	output_layer->gradient_component()->set_delta(&delta);

	BaseLayer* prev_layer = output_layer;

	for (auto it = ++_network->_backward_graph.begin(); it != _network->_backward_graph.end(); ++it) {
		if ((*it)->gradient_component() != nullptr) {
			(*it)->gradient_component()->calc_delta(_network->get_connection((*it)->id(), prev_layer->id())->get_weights(), prev_layer->gradient_component()->get_state());
			prev_layer = *it;
		}		
	}

	for (auto it = _network->_backward_graph.begin(); it != _network->_backward_graph.end(); ++it) {
		if ((*it)->gradient_component() != nullptr) {
			(*it)->gradient_component()->calc_gradient(_w_gradient, _b_gradient);
		}
	}

	for (auto it = _network->_connections.begin(); it != _network->_connections.end(); ++it) {
		if (it->second->is_trainable())
		{
			_w_gradient[(*it).first] = *_network->_layers[it->second->get_in_id()]->get_output() * *_network->_layers[it->second->get_out_id()]->gradient_component()->get_input_delta();
		}		
	}
}

void NetworkGradient::update(map<string, Tensor> &p_update) const {
	for (auto it = _network->_connections.begin(); it != _network->_connections.end(); ++it) {
		if (it->second->is_trainable())
		{
			it->second->update_weights(p_update[it->first]);
		}
	}

	for (auto it = _network->_backward_graph.begin(); it != _network->_backward_graph.end(); ++it) {
		(*it)->update(p_update);
	}
}

void NetworkGradient::check_gradient(Tensor* p_input, Tensor* p_target) {
	const double epsilon = 1e-4;

	for (auto it = _network->_connections.begin(); it != _network->_connections.end(); ++it) {
		if (it->second->is_trainable())
		{
			cout << it->second->get_id() << endl;
			for(int i = 0; i < it->second->get_weights()->size(); i++) {
				const double w = it->second->get_weights()->at(i);
				it->second->get_weights()->set(i, w + epsilon);
				const double Je_plus = check_estimate(p_input, p_target);
				it->second->get_weights()->set(i, w - epsilon);
				const double Je_minus = check_estimate(p_input, p_target);

				it->second->get_weights()->set(i, w);

				const double de = (Je_plus - Je_minus) / (2 * epsilon);

				cout << i << " " << _w_gradient[it->second->get_id()].at(i) - de << endl;
			}
			
		}
	}

	for (auto it = _network->_backward_graph.begin(); it != _network->_backward_graph.end(); ++it) {
		if ((*it)->gradient_component() != nullptr) {
			cout << (*it)->id() << endl;

			for (auto c = (*it)->get_connections()->begin(); c != (*it)->get_connections()->end(); ++c) {
				cout << c->second->get_id() << endl;
				for (int i = 0; i < c->second->get_weights()->size(); i++) {
					const double w = c->second->get_weights()->at(i);
					c->second->get_weights()->set(i, w + epsilon);
					const double Je_plus = check_estimate(p_input, p_target);
					c->second->get_weights()->set(i, w - epsilon);
					const double Je_minus = check_estimate(p_input, p_target);

					c->second->get_weights()->set(i, w);

					const double de = (Je_plus - Je_minus) / (2 * epsilon);

					cout << i << " " << _w_gradient[c->second->get_id()].at(i) - de << endl;
				}
			}

			for (auto g = (*it)->get_groups()->begin(); g != (*it)->get_groups()->end(); ++g) {				
				cout << g->second->get_id() << endl;
				SimpleCellGroup* group = dynamic_cast<SimpleCellGroup*>(g->second);
				if (group != nullptr)
				{
					for (int i = 0; i < group->get_bias()->size(); i++) {
						const double b = group->get_bias()->at(i);
						group->get_bias()->set(i, b + epsilon);
						const double Je_plus = check_estimate(p_input, p_target);
						group->get_bias()->set(i, b - epsilon);
						const double Je_minus = check_estimate(p_input, p_target);

						group->get_bias()->set(i, b);

						const double de = (Je_plus - Je_minus) / (2 * epsilon);

						cout << i << " " << _b_gradient[g->second->get_id()].at(i) - de << endl;
					}
				}
			}
		}
	}
}

double NetworkGradient::check_estimate(Tensor* p_input, Tensor* p_target) const {
	_network->activate(p_input);
	return _cost_function->cost(_network->get_output(), p_target);
}
