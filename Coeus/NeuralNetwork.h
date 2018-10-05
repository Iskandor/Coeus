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
	friend class NetworkGradient;
public:
	NeuralNetwork();
	NeuralNetwork(NeuralNetwork &p_copy);
	virtual ~NeuralNetwork();

	void init();
	virtual void activate(Tensor* p_input);
	virtual void activate(vector<Tensor*>* p_input);
	virtual void override(NeuralNetwork* p_network);

	BaseLayer*	add_layer(BaseLayer* p_layer);
	BaseLayer*	get_layer(const string& p_layer);
	Connection* add_connection(const string& p_input_layer, const string& p_output_layer, Connection::INIT p_init, double p_limit = 0, bool p_trainable = true);
	Connection* get_connection(const string& p_input_layer, const string& p_output_layer);
	vector<BaseLayer*> get_input_layers(const string& p_layer);

	Tensor*		get_output() { return _layers[_output_layer]->get_output(); }	
	vector<Tensor*> get_input();

protected:
	void activate();
	void create_directed_graph();

private:
	void create_param_map(NeuralNetwork* p_network);

	map<string, BaseLayer*> _layers;
	map<string, Connection*> _connections;
	map<string, vector<string>> _graph;
	list<BaseLayer*> _forward_graph;
	list<BaseLayer*> _backward_graph;

	vector<string> _input_layer;
	string _output_layer;

	map<string, string> _param_map;
};

}

