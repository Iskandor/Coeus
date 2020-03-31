#pragma once
#include "dense_layer.h"
#include <map>
#include <vector>
#include <list>
#include "variable.h"

class __declspec(dllexport) neural_network : public param_model
{
public:
	neural_network();
	~neural_network();

	tensor& forward(tensor* p_input);
	tensor& forward(std::map<std::string, tensor*> &p_input);
	tensor& backward(tensor& p_delta);

	dense_layer*	add_layer(dense_layer* p_layer);
	void			add_connection(const std::string& p_input_layer, const std::string& p_output_layer);
	void			init();

private:
	void create_directed_graph();

	std::map<std::string, std::vector<tensor*>> _layer_input_list;
	std::map<std::string, std::vector<tensor>>	_layer_delta_list;

	std::map<std::string, tensor>				_layer_input_tensor;
	std::map<std::string, tensor>				_layer_delta_tensor;
	std::map<std::string, variable>				_input;

	std::map<std::string, dense_layer*> _layers;
	std::map<std::string, std::vector<std::string>> _graph;
	std::list<dense_layer*> _forward_graph;
	std::list<dense_layer*> _backward_graph;

	std::vector<std::string>	_input_layer;
	std::string					_output_layer;
};

