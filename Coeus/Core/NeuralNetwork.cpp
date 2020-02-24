#include "NeuralNetwork.h"
#include <set>
#include <queue>
#include <numeric>
#include "IOUtils.h"
#include "TensorOperator.h"
#include "CoreLayer.h"
#include "RecurrentLayer.h"
#include "LSTMLayer.h"
#include "IDGen.h"
#include "ParamModelStorage.h"


using namespace Coeus;

/**
 * \brief Creates empty network
 */
NeuralNetwork::NeuralNetwork() : ParamModel()
{
	ParamModelStorage::instance().bind(_id, _id);
}

/**
 * \brief Creates network from data in json format loaded from the file
 * \param p_data text file in json format
 */
NeuralNetwork::NeuralNetwork(json p_data)
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

/**
 * \brief Creates a copy or clone of the network. The copy shares the parameters with the original and every change will have impact on the original as well as on the copy.
 * The clone has its own parameters with the same value as the original and change in parameters of one network will leave the others network parameters intact.
 * \param p_copy source network
 * \param p_clone copy / clone flag 
 */
NeuralNetwork::NeuralNetwork(NeuralNetwork& p_copy, const bool p_clone) : ParamModel()
{
	if (p_clone)
	{
		ParamModelStorage::instance().bind(_id, _id);
	}
	else
	{
		ParamModelStorage::instance().bind(p_copy._id, _id);
	}

	for(auto l : p_copy._layers)
	{
		add_layer(l.second->copy(p_clone));
	}

	_graph = p_copy._graph;
	
	init();
}


NeuralNetwork::~NeuralNetwork()
{
	ParamModelStorage::instance().release(this);

	for (const auto& layer : _layers)
	{
		delete layer.second;
	}
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
		if (it->second->get_in_dim() != 0)
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

	ParamModelStorage::instance().add(_id, this);
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

void NeuralNetwork::reset()
{
	for (auto& _layer : _layers)
	{
		_layer.second->reset();
	}
}

/*
void NeuralNetwork::copy_params(const NeuralNetwork* p_model)
{
	if (ParamModelStorage::instance().is_bound(_id))
	{
		assert(0, "NeuralNetwork::copy_params : This model is bound");
	}

	for (const auto& layer : p_model->_layers)
	{
		_layers[layer.first]->copy_params(layer.second);
	}
}
*/

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


