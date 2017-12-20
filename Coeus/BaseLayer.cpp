#include "BaseLayer.h"
#include <set>

using namespace Coeus;

BaseLayer::BaseLayer()
{
}


BaseLayer::~BaseLayer()
{
	for(auto it = _groups.begin(); it != _groups.end(); ++it) {
		delete (*it).second;
	}

	for (auto it = _connections.begin(); it != _connections.end(); ++it) {
		delete (*it).second;
	}
}

NeuralGroup* BaseLayer::add_group(const int p_dim, const NeuralGroup::ACTIVATION p_activation, const bool p_bias) {
	NeuralGroup* g = new NeuralGroup(p_dim, p_activation, p_bias);
	_groups[g->getId()] = g;

	return g;
}

void BaseLayer::add_connection(NeuralGroup* p_inGroup, NeuralGroup* p_outGroup, const Connection::INIT p_init, const double p_limit) {
	Connection* c = new Connection(p_inGroup->getDim(), p_outGroup->getDim(), p_inGroup->getId(), p_outGroup->getId());
	c->init(p_init, p_limit);

	_connections[c->get_id()] = c;

	_graph[p_outGroup->getId()].push_back(p_inGroup->getId());

	set<string> controll_set;

	for(auto it = _graph.begin(); it != _graph.end(); ++it) {
		if ((*it).second.empty()) {
			_inputGroup = (*it).first;
		}
		else {
			for (auto ag = (*it).second.begin(); ag != (*it).second.end(); ++ag) {
				controll_set.insert(*ag);
			}
		}
	}

	for (auto it = _graph.begin(); it != _graph.end(); ++it) {
		if (controll_set.find((*it).first) == controll_set.end()) {
			_outputGroup = (*it).first;
		}
	}
}

Connection* BaseLayer::get_connection(const string p_input_group, const string p_output_group) {
	Connection* result = nullptr;

	for (auto it = _connections.begin(); it != _connections.end(); ++it) {
		if ((*it).second->get_in_id().compare(p_input_group) == 0 && (*it).second->get_out_id().compare(p_output_group) == 0) {
			result = (*it).second;
		}
	}

	return result;
}
