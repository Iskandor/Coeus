#include "NeuralNetwork.h"
#include <set>
#include <queue>


using namespace Coeus;

NeuralNetwork::NeuralNetwork() = default;


NeuralNetwork::~NeuralNetwork()
{
	for (auto it = _layers.begin(); it != _layers.end(); ++it) {
		delete (*it).second;
	}

	for (auto it = _connections.begin(); it != _connections.end(); ++it) {
		delete (*it).second;
	}
}

void NeuralNetwork::activate(Tensor * p_input)
{
	_layers[_input_layer]->activate(p_input);

	for (auto layer = _forward_graph.begin(); layer != _forward_graph.end(); ++layer) {
		for (auto input = _graph[(*layer)->id()].begin(); input != _graph[(*layer)->id()].end(); ++input) {
			(*layer)->integrate(_layers[*input]->get_output(), _connections[(*layer)->id() + "_" + (*input)]->get_weights());
		}
		(*layer)->activate();
	}
}

BaseLayer* NeuralNetwork::add_layer(BaseLayer* p_layer) {
	_layers[p_layer->id()] = p_layer;

	return p_layer;
}

Connection* NeuralNetwork::add_connection(const string& p_input_layer, const string& p_output_layer, const Connection::INIT p_init, const double p_limit) {
	BaseLayer* in_layer = _layers[p_input_layer];
	BaseLayer* out_layer = _layers[p_output_layer];

	Connection* c = new Connection(in_layer->output_dim(), out_layer->output_dim(), in_layer->id(), out_layer->id());
	c->init(p_init, p_limit);

	_connections[c->get_id()] = c;

	_graph[out_layer->id()].push_back(in_layer->id());

	set<string> controll_set;

	for (auto it = _layers.begin(); it != _layers.end(); ++it) {
		if (_graph.find(it->first) == _graph.end()) {
			_input_layer = it->first;
		}
		else {
			for (auto ag = _graph[it->first].begin(); ag != _graph[it->first].end(); ++ag) {
				controll_set.insert(*ag);
			}
		}
	}

	for (auto it = _graph.begin(); it != _graph.end(); ++it) {
		if (controll_set.find((*it).first) == controll_set.end()) {
			_output_layer = (*it).first;
		}
	}

	create_directed_graph();

	out_layer->post_connection(in_layer);

	return c;
}

Connection* NeuralNetwork::get_connection(const string& p_input_layer, const string& p_output_layer) {
	Connection* result = nullptr;

	for (auto it = _connections.begin(); it != _connections.end(); ++it) {
		if ((*it).second->get_in_id() == p_input_layer && (*it).second->get_out_id() == p_output_layer) {
			result = (*it).second;
		}
	}

	return result;
}

void NeuralNetwork::create_directed_graph()
{
	_forward_graph.clear();
	_backward_graph.clear();

	for(auto it = _layers.begin(); it != _layers.end(); ++it)
	{
		it->second->set_valid(false);
	}

	queue<string> q;

	q.push(_input_layer);

	while (!q.empty())
	{
		const string v = q.front();
		q.pop();

		if (!_layers[v]->is_valid()) {
			_forward_graph.push_back(_layers[v]);
			_layers[v]->set_valid(true);

			for (auto it = _graph.begin(); it != _graph.end(); ++it) {
				if (!_layers[it->first]->is_valid()) {
					for (auto n = it->second.begin(); n != it->second.end(); ++n) {
						if (*n == v) {
							q.push(it->first);
						}
					}
				}
			}
		}
	}

	for (auto it = _forward_graph.rbegin(); it != _forward_graph.rend(); ++it) {
		_backward_graph.push_back(*it);
	}
}
