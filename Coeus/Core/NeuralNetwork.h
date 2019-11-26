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
	friend class NaturalGradient;
public:
	NeuralNetwork();
	explicit NeuralNetwork(json p_data);
	NeuralNetwork(NeuralNetwork &p_copy, bool p_clone = false);
	virtual ~NeuralNetwork();

	void init();
	virtual void activate(Tensor* p_input);
	virtual void activate(vector<Tensor*>* p_input);
	void reset();
	void copy_params(const NeuralNetwork* p_model);

	BaseLayer*	add_layer(BaseLayer* p_layer);
	BaseLayer*	get_layer(const string& p_layer);
	void		add_connection(const string& p_input_layer, const string& p_output_layer);
	vector<BaseLayer*> get_input_layers(const string& p_layer);

	Tensor*		get_output() { return _layers[_output_layer]->get_output(); }	
	vector<Tensor*> get_input();

	int get_output_dim();
	int get_input_dim();

	json get_json() const;

protected:	
	void activate();	
	void create_directed_graph();

private:
	map<string, BaseLayer*> _layers;
	map<string, vector<string>> _graph;
	list<BaseLayer*> _forward_graph;
	list<BaseLayer*> _backward_graph;

	vector<string>	_input_layer;
	string			_output_layer;
};

}

