#include "NeuralNetwork.h"
#include <set>


using namespace Coeus;

NeuralNetwork::NeuralNetwork()
{
}


NeuralNetwork::~NeuralNetwork()
{
	for (auto it = _layers.begin(); it != _layers.end(); ++it) {
		delete (*it).second;
	}

	for (auto it = _connections.begin(); it != _connections.end(); ++it) {
		delete (*it).second;
	}
}

BaseLayer* NeuralNetwork::add_layer(BaseLayer* p_layer) {
	_layers[p_layer->id()] = p_layer;

	return p_layer;
}

Connection* NeuralNetwork::add_connection(BaseLayer* p_inGroup, BaseLayer* p_outGroup, const Connection::INIT p_init, const double p_limit) {
	Connection* c = new Connection(p_inGroup->input_dim(), p_outGroup->output_dim(), p_inGroup->id(), p_outGroup->id());
	c->init(p_init, p_limit);

	_connections[c->get_id()] = c;

	_graph[p_outGroup->id()].push_back(p_inGroup->id());

	set<string> controll_set;

	for (auto it = _layers.begin(); it != _layers.end(); ++it) {
		if (_graph.find(it->first) == _graph.end()) {
			_inputLayer = it->first;
		}
		else {
			for (auto ag = _graph[it->first].begin(); ag != _graph[it->first].end(); ++ag) {
				controll_set.insert(*ag);
			}
		}
	}

	for (auto it = _graph.begin(); it != _graph.end(); ++it) {
		if (controll_set.find((*it).first) == controll_set.end()) {
			_outputLayer = (*it).first;
		}
	}

	return c;
}

Connection* NeuralNetwork::get_connection(const string p_input_group, const string p_output_group) {
	Connection* result = nullptr;

	for (auto it = _connections.begin(); it != _connections.end(); ++it) {
		if ((*it).second->get_in_id().compare(p_input_group) == 0 && (*it).second->get_out_id().compare(p_output_group) == 0) {
			result = (*it).second;
		}
	}

	return result;
}
