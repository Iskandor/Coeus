#pragma once
#include <string>
#include <map>
#include <list>
#include "BaseLayer.h"

using namespace std;
using namespace nlohmann;

namespace Coeus {

class __declspec(dllexport) NeuralNetwork : public ParamModel
{
	friend class NetworkGradient;
public:
	NeuralNetwork();
	explicit NeuralNetwork(json& p_data);
	NeuralNetwork(NeuralNetwork &p_copy);
	virtual ~NeuralNetwork();
	NeuralNetwork* clone() const;

	void init();
	virtual void activate(Tensor* p_input);
	virtual void activate(vector<Tensor*>* p_input);
	virtual void override(NeuralNetwork* p_network);
	virtual void update(map<string, Tensor> *p_update) const;
	void reset();

	BaseLayer*	add_layer(BaseLayer* p_layer);
	BaseLayer*	get_layer(const string& p_layer);
	void add_connection(const string& p_input_layer, const string& p_output_layer);
	vector<BaseLayer*> get_input_layers(const string& p_layer);

	Tensor*		get_output() { return _layers[_output_layer]->get_output(); }	
	vector<Tensor*> get_input();

	json get_json() const;

protected:	
	void activate();	
	void create_directed_graph();

private:	
	void create_param_map(NeuralNetwork* p_network);

	map<string, BaseLayer*> _layers;
	map<string, vector<string>> _graph;
	list<BaseLayer*> _forward_graph;
	list<BaseLayer*> _backward_graph;

	vector<string>	_input_layer;
	string			_output_layer;

	map<string, string> _param_map;
};

}

