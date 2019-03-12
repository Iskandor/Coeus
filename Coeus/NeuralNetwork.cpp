#include "NeuralNetwork.h"
#include <set>
#include <queue>
#include "InputLayer.h"
#include "CoreLayer.h"
#include "RecurrentLayer.h"
#include <numeric>
#include "IOUtils.h"


using namespace Coeus;

NeuralNetwork::NeuralNetwork() = default;

NeuralNetwork::NeuralNetwork(json p_data)
{
	for (json::iterator it = p_data["layers"].begin(); it != p_data["layers"].end(); ++it) {
		add_layer(IOUtils::create_layer(it.value()));
	}

	for (json::iterator it = p_data["connections"].begin(); it != p_data["connections"].end(); ++it) {
		add_connection(new Connection(it.value()));
	}

	init();
}

NeuralNetwork::NeuralNetwork(NeuralNetwork& p_copy) {
	_param_map.clear();

	BaseLayer* layer = nullptr;

	for (auto it = p_copy._layers.begin(); it != p_copy._layers.end(); ++it) {
		switch ((*it).second->get_type()) {

			case BaseLayer::SOM: 

			break;
			case BaseLayer::MSOM: 

			break;
			case BaseLayer::INPUT: 
				layer = add_layer(new InputLayer(*dynamic_cast<InputLayer*>((*it).second)));
			break;
			case BaseLayer::CORE: 
				layer = add_layer(new CoreLayer(*dynamic_cast<CoreLayer*>((*it).second)));
			break;
			case BaseLayer::RECURRENT: 
				layer = add_layer(new RecurrentLayer(*dynamic_cast<RecurrentLayer*>((*it).second)));
			break;
			case BaseLayer::LSTM: 
			break;
			case BaseLayer::LSOM: 
			break;
			default: ;
		}

		_param_map[(*it).second->get_id()] = layer->get_id();
	}
	Connection* connection = nullptr;

	for (auto out = p_copy._graph.begin(); out != p_copy._graph.end(); ++out) {
		for(auto in = out->second.begin(); in != out->second.end(); ++in) {
			connection = add_connection(_param_map[*in], _param_map[out->first], Connection::UNIFORM, true);
			connection->override(p_copy.get_connection(*in, out->first));
		}
	}

	_param_map.clear();
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

NeuralNetwork* NeuralNetwork::clone() const
{
	NeuralNetwork* result = new NeuralNetwork();

	for (auto it = _layers.begin(); it != _layers.end(); ++it) {
		result->add_layer((*it).second->clone());
	}

	for (auto it = _connections.begin(); it != _connections.end(); ++it) {
		result->add_connection(it->second->clone());
	}

	result->init();

	return result;
}

void NeuralNetwork::init()
{
	set<string> control_set;

	for (auto it = _layers.begin(); it != _layers.end(); ++it) {
		if (_graph.find(it->first) != _graph.end()) {
			for (auto ag = _graph[it->first].begin(); ag != _graph[it->first].end(); ++ag) {
				control_set.insert(*ag);
			}
		}
	}

	for (auto it = _graph.begin(); it != _graph.end(); ++it) {
		if (control_set.find((*it).first) == control_set.end()) {
			_output_layer = (*it).first;
		}
	}

	create_directed_graph();

	for (auto it = _graph.begin(); it != _graph.end(); ++it) {
		vector<BaseLayer*> input;

		for (auto n = it->second.begin(); n != it->second.end(); ++n) {
			input.push_back(_layers[*n]);
		}

		_layers[it->first]->init(input);
		add_param(_layers[it->first]);
	}
}

void NeuralNetwork::activate(Tensor * p_input)
{
	// single input
	if (p_input->rank() == 1)
	{
		_layers[_input_layer[0]]->activate(p_input);

		activate();
	}

	// sequence
	if (p_input->rank() == 2)
	{
		Tensor input = Tensor::Zero({ p_input->shape(1) });

		reset();
		for (int i = 0; i < p_input->shape(0); i++)
		{
			p_input->get_row(input, i);
			_layers[_input_layer[0]]->activate(&input);

			activate();
		}
	}
}

void NeuralNetwork::activate(vector<Tensor*>* p_input)
{
	if (p_input->size() != _input_layer.size())
	{
		assert(("Sequence size not equal to input layers count", 0));
	}

	for(unsigned i = 0; i < _input_layer.size(); i++)
	{
		_layers[_input_layer[i]]->activate(p_input->at(i));
	}

	activate();
}

void NeuralNetwork::override(NeuralNetwork* p_network) {

	bool param_map_init = true;

	for (auto it = _layers.begin(); it != _layers.end(); ++it) {
		if (_param_map.count(it->first) == 0) {
			param_map_init = false;
			break;
		}
	}

	if (param_map_init) {
		for (auto it = _connections.begin(); it != _connections.end(); ++it) {
			if (_param_map.count(it->first) == 0) {
				param_map_init = false;
				break;
			}
		}
	}

	if (!param_map_init) {
		create_param_map(p_network);
	}

	for(auto it = _layers.begin(); it != _layers.end(); ++it) {
		_layers[it->first]->override(p_network->_layers[_param_map[it->first]]);
	}

	for (auto it = _connections.begin(); it != _connections.end(); ++it) {
		_connections[it->first]->override(p_network->_connections[_param_map[it->first]]);
	}
}

void NeuralNetwork::update(map<string, Tensor>* p_update) const
{
	for (auto it = _params.begin(); it != _params.end(); ++it) {
		*it->second += (*p_update)[it->first];
	}
}

void NeuralNetwork::reset()
{
	for (auto& _layer : _layers)
	{
		_layer.second->reset();
	}
}

vector<Tensor*> NeuralNetwork::get_input() {
	vector<Tensor*> result;

	for(auto it = _input_layer.begin(); it != _input_layer.end(); ++it) {
		result.push_back(_layers[*it]->get_output());
	}

	return vector<Tensor*>(result);
}

void NeuralNetwork::activate()
{
	for (auto layer = _forward_graph.begin(); layer != _forward_graph.end(); ++layer) {
		for (auto input = _graph[(*layer)->get_id()].begin(); input != _graph[(*layer)->get_id()].end(); ++input) {
			(*layer)->integrate(_layers[*input]->get_output(), _connections[(*layer)->get_id() + "_" + (*input)]->get_weights());
		}
		(*layer)->activate();
	}
}

BaseLayer* NeuralNetwork::add_layer(BaseLayer* p_layer) {
	_layers[p_layer->get_id()] = p_layer;

	if (dynamic_cast<InputLayer*>(p_layer) != nullptr)
	{
		_input_layer.push_back(p_layer->get_id());
	}

	return p_layer;
}

BaseLayer* NeuralNetwork::get_layer(const string& p_layer) {
	return _layers[p_layer];
}

Connection* NeuralNetwork::add_connection(const string& p_input_layer, const string& p_output_layer, const Connection::INIT p_init, const float p_arg1, const float p_arg2) {
	BaseLayer* in_layer = _layers[p_input_layer];
	BaseLayer* out_layer = _layers[p_output_layer];

	Connection* c = new Connection(in_layer->output_dim(), out_layer->output_dim(), in_layer->get_id(), out_layer->get_id(), p_init, true, p_arg1, p_arg2);

	_connections[c->get_id()] = c;

	_graph[out_layer->get_id()].push_back(in_layer->get_id());

	add_param(c);

	return c;
}

void NeuralNetwork::add_connection(Connection* p_connection)
{
	_connections[p_connection->get_id()] = p_connection;
	_graph[p_connection->get_out_id()].push_back(p_connection->get_in_id());
	if (p_connection->is_trainable()) add_param(p_connection);
}

Connection* NeuralNetwork::get_connection(const string& p_input_layer, const string& p_output_layer) {
	Connection* result = nullptr;

	for (auto it = _connections.begin(); it != _connections.end(); ++it) {
		if ((*it).second->get_in_id() == p_input_layer && (*it).second->get_out_id() == p_output_layer) {
			result = (*it).second;
		}
	}

	return result;
}

vector<BaseLayer*> NeuralNetwork::get_input_layers(const string& p_layer) {
	vector<BaseLayer*> result;

	for(auto it = _graph[p_layer].begin(); it != _graph[p_layer].end(); ++it) {
		result.push_back(_layers[*it]);
	}

	return result;
}

void NeuralNetwork::create_directed_graph()
{
	_forward_graph.clear();
	_backward_graph.clear();

	for(auto it = _layers.begin(); it != _layers.end(); ++it)
	{
		it->second->set_valid(false);
	}

	queue<string> q;

	for(auto it = _input_layer.begin(); it != _input_layer.end(); ++it)
	{
		q.push(*it);
	}
	

	while (!q.empty())
	{
		const string v = q.front();
		q.pop();

		if (!_layers[v]->is_valid()) {
			_forward_graph.push_back(_layers[v]);
			_layers[v]->set_valid(true);

			for (auto it = _graph.begin(); it != _graph.end(); ++it) {
				if (!_layers[it->first]->is_valid()) {
					for (auto n = it->second.begin(); n != it->second.end(); ++n) {
						if (*n == v) {
							q.push(it->first);
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

void NeuralNetwork::create_param_map(NeuralNetwork* p_network) {
	_param_map.clear();

	vector<string> target;

	for (auto it = _forward_graph.begin(); it != _forward_graph.end(); ++it) {
		target.push_back((*it)->get_id());
	}

	vector<string> source;

	for (auto it = p_network->_forward_graph.begin(); it != p_network->_forward_graph.end(); ++it) {
		source.push_back((*it)->get_id());
	}

	if (target.size() == source.size()) {
		for (int i = 0; i < source.size(); i++) {
			_param_map[target[i]] = source[i];
		}
	}
	else {
		//TODO warning inequal sizes
	}

	for(auto it = _graph.begin(); it != _graph.end(); ++it) {
		for(auto c = (*it).second.begin(); c != (*it).second.end(); ++c) {
			_param_map[(*c + "_" + (*it).first)] = _param_map[(*c)] + "_" + _param_map[(*it).first];
		}
	}
}

json NeuralNetwork::get_json() const
{
	json data;

	for (auto it = _layers.begin(); it != _layers.end(); ++it) {
		data["layers"][it->first] = it->second->get_json();
	}

	for (auto it = _connections.begin(); it != _connections.end(); ++it) {
		data["connections"][it->first] = it->second->get_json();
	}

	return data;
}


