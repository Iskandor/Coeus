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
	std::map<std::string, tensor*>& backward(tensor& p_delta);

	dense_layer*	add_layer(dense_layer* p_layer);
	void			add_connection(const std::string& p_input_layer, const std::string& p_output_layer);
	void			init();

private:
	struct layer_variable
	{
		tensor					input;
		tensor					delta;
		std::vector<tensor*>	input_list;
		std::vector<tensor*>	delta_list;
	};

	void create_directed_graph();

	std::map<std::string, layer_variable>		_layer_variables;
	std::map<std::string, variable>				_input;
	std::map<std::string, tensor*>				_delta;

	std::map<std::string, dense_layer*> _layers;
	std::map<std::string, std::vector<std::string>> _graph;
	std::list<dense_layer*> _forward_graph;
	std::list<dense_layer*> _backward_graph;

	std::vector<std::string>	_input_layer;
	std::string					_output_layer;
};

