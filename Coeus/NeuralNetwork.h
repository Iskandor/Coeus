#pragma once
#include <string>
#include <map>
#include <list>
#include "BaseLayer.h"
#include "Connection.h"

using namespace std;

namespace Coeus {

class __declspec(dllexport) NeuralNetwork
{
public:
	NeuralNetwork();
	virtual ~NeuralNetwork();

	virtual void activate(Tensor* p_input);

	BaseLayer*	add_layer(BaseLayer* p_layer);
	Connection* add_connection(string p_input_layer, string p_output_layer, Connection::INIT p_init, double p_limit);
	Connection* get_connection(string p_input_layer, string p_output_layer);

protected:
	void create_directed_graph();

	map<string, BaseLayer*> _layers;
	map<string, Connection*> _connections;
	map<string, vector<string>> _graph;
	list<BaseLayer*> _forward_graph;
	list<BaseLayer*> _backward_graph;

	string _input_layer;
	string _output_layer;
};

}

