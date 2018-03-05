#pragma once
#include <string>
#include <map>
#include "BaseLayer.h"
#include "Connection.h"

using namespace std;

namespace Coeus {

class __declspec(dllexport) NeuralNetwork
{
public:
	NeuralNetwork();
	virtual ~NeuralNetwork();

	virtual void activate(Tensor* p_input) = 0;

protected:
	BaseLayer* add_layer(BaseLayer* p_layer);
	Connection* add_connection(BaseLayer* p_in_group, BaseLayer* p_out_group, Connection::INIT p_init, double p_limit);

	Connection* get_connection(string p_input_group, string p_output_group);

	map<string, BaseLayer*> _layers;
	map<string, Connection*> _connections;
	map<string, vector<string>> _graph;

	string _inputLayer;
	string _outputLayer;
};

}

