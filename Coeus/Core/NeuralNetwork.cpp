#include "NeuralNetwork.h"
#include <set>
#include <queue>
#include <numeric>
#include "IOUtils.h"
#include "TensorOperator.h"
#include "CoreLayer.h"
#include "RecurrentLayer.h"
#include "LSTMLayer.h"


using namespace Coeus;

NeuralNetwork::NeuralNetwork() :
	_own_model(true)
{		
}

NeuralNetwork::NeuralNetwork(json p_data) :
	_own_model(true)
{
	for (json::iterator it = p_data["layers"].begin(); it != p_data["layers"].end(); ++it) {
		add_layer(IOUtils::create_layer(it.value()));
	}

	for (json::iterator out_layer = p_data["graph"].begin(); out_layer != p_data["graph"].end(); ++out_layer) {
		for (json::iterator in_layer = out_layer.value().begin(); in_layer != out_layer.value().end(); ++in_layer) {
			add_connection(in_layer.value().get<string>(), out_layer.key());
		}
	}

	init();
}

NeuralNetwork::NeuralNetwork(NeuralNetwork& p_copy) :
	_own_model(false)
{
	_param_map.clear();

	BaseLayer* layer = nullptr;

	for (auto it = p_copy._layers.begin(); it != p_copy._layers.end(); ++it) {
		switch ((*it).second->get_type()) {

			case BaseLayer::SOM: 

			break;
			case BaseLayer::MSOM: 

			break;
			case BaseLayer::CORE: 
				layer = add_layer(new CoreLayer(*dynamic_cast<CoreLayer*>((*it).second)));
			break;
			case BaseLayer::RECURRENT: 
				layer = add_layer(new RecurrentLayer(*dynamic_cast<RecurrentLayer*>((*it).second)));
			break;
			case BaseLayer::LSTM:
				layer = add_layer(new LSTMLayer(*dynamic_cast<LSTMLayer*>((*it).second)));
			break;
			case BaseLayer::LSOM: 
			break;
			default: ;
		}

		_param_map[(*it).second->get_id()] = layer->get_id();
	}

	for (auto out = p_copy._graph.begin(); out != p_copy._graph.end(); ++out) {
		for(auto in = out->second.begin(); in != out->second.end(); ++in) {
			add_connection(_param_map[*in], _param_map[out->first]);
		}
	}

	_param_map.clear();

	init();
}


NeuralNetwork::~NeuralNetwork()
{
	for (const auto& layer : _layers)
	{
		delete layer.second;
	}

	for (const auto& param : _params)
	{
		delete param.second;
	}
}

NeuralNetwork* NeuralNetwork::clone() const
{
	NeuralNetwork* result = new NeuralNetwork();
	result->_own_model = true;

	for (auto it = _layers.begin(); it != _layers.end(); ++it) {
		result->add_layer((*it).second->clone());
	}

	result->_graph = _graph;

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

	for (auto it = _layers.begin(); it != _layers.end(); ++it) {
		if (_graph.find(it->first) == _graph.end() || _graph[it->first].empty())
		{
			_input_layer.push_back(it->first);
		}
	}

	create_directed_graph();

	for (auto layer = _forward_graph.begin(); layer != _forward_graph.end(); ++layer) {
		vector<BaseLayer*> input;
		vector<BaseLayer*> output;

		for (auto n = _graph[(*layer)->get_id()].begin(); n != _graph[(*layer)->get_id()].end(); ++n) {
			input.push_back(_layers[*n]);
		}
		for (auto o = _graph.begin(); o != _graph.end(); ++o) {
			for(auto i = _graph[o->first].begin(); i != _graph[o->first].end(); ++i)
			{
				if (*i == (*layer)->get_id()) 
				{
					output.push_back(_layers[o->first]);
				}
			}
		}

		_layers[(*layer)->get_id()]->init(input,output);
		add_param(_layers[(*layer)->get_id()]);
	}
}

void NeuralNetwork::activate(Tensor* p_input)
{
	// single input
	_layers[_input_layer[0]]->integrate(p_input);
	activate();
}

void NeuralNetwork::activate(vector<Tensor*>* p_input)
{
	// sequence
	reset();
	for (auto& it : *p_input)
	{
		_layers[_input_layer[0]]->integrate(it);
		activate();
	}
}

void NeuralNetwork::override(NeuralNetwork* p_network) {

	bool param_map_init = true;

	for (auto it = _layers.begin(); it != _layers.end(); ++it) {
		if (_param_map.count(it->first) == 0) {
			param_map_init = false;
			break;
		}
	}

	if (!param_map_init) {
		create_param_map(p_network);
	}

	for(auto it = _layers.begin(); it != _layers.end(); ++it) {
		_layers[it->first]->override(p_network->_layers[_param_map[it->first]]);
	}
}

void NeuralNetwork::update(map<string, Tensor>* p_update) const
{
	for (const auto& param : _params)
	{
		TensorOperator::instance().vv_add(param.second->arr(), (*p_update)[param.first].arr(), param.second->arr(), param.second->size());
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

	for (auto& it : _input_layer)
	{
		result.push_back(_layers[it]->get_output());
	}

	return vector<Tensor*>(result);
}

int NeuralNetwork::get_output_dim()
{
	return _layers[_output_layer]->get_dim();
}

int NeuralNetwork::get_input_dim()
{
	int dim = 0;
	for (auto& it : _input_layer)
	{
		dim += _layers[it]->get_in_dim();
	}

	return dim;
}

void NeuralNetwork::activate()
{
	for (auto& layer : _forward_graph)
	{
		for (auto input = _graph[layer->get_id()].begin(); input != _graph[layer->get_id()].end(); ++input) {
			layer->integrate(_layers[*input]->get_output());
		}
		layer->activate();
	}
}

BaseLayer* NeuralNetwork::add_layer(BaseLayer* p_layer) {
	_layers[p_layer->get_id()] = p_layer;

	return p_layer;
}

BaseLayer* NeuralNetwork::get_layer(const string& p_layer) {
	return _layers[p_layer];
}

void NeuralNetwork::add_connection(const string& p_input_layer, const string& p_output_layer) {
	BaseLayer* in_layer = _layers[p_input_layer];
	BaseLayer* out_layer = _layers[p_output_layer];

	if (in_layer != nullptr && out_layer != nullptr)
	{
		_graph[out_layer->get_id()].push_back(in_layer->get_id());
	}
	else
	{
		assert(("add_connection: One or both layers do not exist", 0));
	}
}

vector<BaseLayer*> NeuralNetwork::get_input_layers(const string& p_layer) {
	vector<BaseLayer*> result;

	for (auto& it : _graph[p_layer])
	{
		result.push_back(_layers[it]);
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

	for (auto& it : _input_layer)
	{
		q.push(it);
	}
	

	while (!q.empty())
	{
		const string v = q.front();
		q.pop();

		if (!_layers[v]->is_valid()) {
			_forward_graph.push_back(_layers[v]);
			_layers[v]->set_valid(true);

			for (auto& it : _graph)
			{
				if (!_layers[it.first]->is_valid()) {
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

	for(const auto& out_layer : _graph)
	{
		data["graph"][out_layer.first] = out_layer.second;
	}

	return data;
}


