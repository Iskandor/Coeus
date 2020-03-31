#include "neural_network.h"
#include <queue>
#include <cassert>
#include <set>


neural_network::neural_network() : param_model()
{
}


neural_network::~neural_network()
{
	for(auto layer : _layers)
	{
		delete layer.second;
	}
}

tensor& neural_network::forward(tensor* p_input)
{
	for (const auto& layer : _input_layer)
	{
		_layers[layer]->forward(*p_input);
	}

	for(auto layer : _forward_graph)
	{
		if (_graph[layer->id()].size() == 1)
		{
			layer->forward(*_layers[_graph[layer->id()][0]]->output());
		}
		else if (_graph[layer->id()].size() > 1)
		{
			tensor::concat(_layer_input_list[layer->id()], _layer_input_tensor[layer->id()]);
			layer->forward(_layer_input_tensor[layer->id()]);
		}
	}

	return *_layers[_output_layer]->output();
}

tensor& neural_network::forward(std::map<std::string, tensor*>& p_input)
{
	for (const auto& layer : _input_layer)
	{
		_input[layer].value() = *p_input[layer];
	}

	for (auto layer : _forward_graph)
	{
		if (_layer_input_list[layer->id()].size() <= 1)
		{
			layer->forward(*_layer_input_list[layer->id()][0]);
		}
		else if (_layer_input_list[layer->id()].size() > 1)
		{
			tensor::concat(_layer_input_list[layer->id()], _layer_input_tensor[layer->id()]);
			layer->forward(_layer_input_tensor[layer->id()]);
		}
	}

	return *_layers[_output_layer]->output();
}


tensor& neural_network::backward(tensor& p_delta)
{
	tensor& delta = p_delta;

	for (auto layer : _backward_graph)
	{
		delta = layer->backward(delta);
		if (_layer_input_list[layer->id()].size() > 1)
		{
			//tensor::split(delta, _layer_delta_list);
		}
		else
		{
			
		}
	}

	return delta;
}

dense_layer* neural_network::add_layer(dense_layer* p_layer)
{
	_layers[p_layer->id()] = p_layer;
	_graph[p_layer->id()] = {};
	return p_layer;
}

void neural_network::add_connection(const std::string& p_input_layer, const std::string& p_output_layer)
{
	dense_layer* in_layer = _layers[p_input_layer];
	dense_layer* out_layer = _layers[p_output_layer];

	if (in_layer != nullptr && out_layer != nullptr)
	{
		_graph[out_layer->id()].push_back(in_layer->id());
	}
	else
	{
		assert(("add_connection: One or both layers do not exist", 0));
	}
}

void neural_network::init()
{
	std::set<std::string> control_set;
	_input_layer.clear();

	for (auto it = _layers.begin(); it != _layers.end(); ++it) {
		if (_graph.find(it->first) != _graph.end()) {
			for (auto ag = _graph[it->first].begin(); ag != _graph[it->first].end(); ++ag) {
				control_set.insert(*ag);
			}
		}
	}

	if (_graph.empty())
	{
		_output_layer = (*_layers.begin()).first;
	}
	else
	{
		for (auto it = _graph.begin(); it != _graph.end(); ++it) {
			if (control_set.find((*it).first) == control_set.end()) {
				_output_layer = (*it).first;
			}
		}
	}

	for (auto& layer : _layers)
	{
		if (layer.second->input_dim() > 0)
		{
			_input_layer.push_back(layer.first);
		}
	}

	create_directed_graph();

	_layer_input_list.clear();
	_layer_input_tensor.clear();

	for (auto layer = _forward_graph.begin(); layer != _forward_graph.end(); ++layer) {
		std::vector<dense_layer*> input;
		//std::vector<dense_layer*> output;

		for (auto n = _graph[(*layer)->id()].begin(); n != _graph[(*layer)->id()].end(); ++n) {
			input.push_back(_layers[*n]);
			_layer_input_list[(*layer)->id()].push_back(_layers[*n]->output());
			_layer_delta_list[(*layer)->id()].push_back(tensor({ 1,_layers[*n]->dim() }));
		}

		if (_input.find((*layer)->id()) != _input.end())
		{
			_layer_input_list[(*layer)->id()].push_back(&_input[(*layer)->id()].value());
			_layer_delta_list[(*layer)->id()].push_back(tensor({ 1,(*layer)->input_dim() }));
		}
		/*
		for (auto o = _graph.begin(); o != _graph.end(); ++o) {
			for (auto i = _graph[o->first].begin(); i != _graph[o->first].end(); ++i)
			{
				if (*i == (*layer)->id())
				{
					output.push_back(_layers[o->first]);
				}
			}
		}
		*/

		_layers[(*layer)->id()]->init(this, input);
		_layer_input_tensor[(*layer)->id()] = tensor::zero({ 1,1 });
	}
}

void neural_network::create_directed_graph()
{
	_forward_graph.clear();
	_backward_graph.clear();

	std::map<std::string, bool> valid;

	for (auto& layer : _layers)
	{
		valid[layer.second->id()] = false;
	}

	std::queue<std::string> q;

	for (auto& it : _input_layer)
	{
		q.push(it);
	}


	while (!q.empty())
	{
		const std::string v = q.front();
		q.pop();

		if (!valid[v]) {
			_forward_graph.push_back(_layers[v]);
			valid[v] = true;

			for (auto& it : _graph)
			{
				if (!valid[it.first]) {
					for (auto n = it.second.begin(); n != it.second.end(); ++n) {
						if (*n == v) {
							q.push(it.first);
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
