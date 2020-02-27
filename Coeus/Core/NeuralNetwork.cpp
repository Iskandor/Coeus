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

/**
 * \brief Initialization of neural network model. Creates directed graph of layers (forward and backward), identifies input and output layers and populates parametric model class
 */
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

/**
 * \brief Activate network on the input (single, minibatch, batch). Used in feed-froward architectures.
 * \param p_input input tensor
 */
void NeuralNetwork::activate(Tensor* p_input)
{
	_layers[_input_layer[0]]->integrate(p_input);
	activate();
}

/**
 * \brief Activate network on the input sequence. Used in recurrent architectures. 
 * \param p_input input sequence of tensors
 */
void NeuralNetwork::activate(vector<Tensor*>* p_input)
{
	reset();
	for (auto& it : *p_input)
	{
		_layers[_input_layer[0]]->integrate(it);
		activate();
	}
}

/**
 * \brief Activate network with more input layers on the input, where the keys of the map are names of the input layers
 * \param p_input map of input tensors
 */
void NeuralNetwork::activate(map<string, Tensor*>& p_input)
{
	for (auto& it : p_input)
	{
		_layers[it.first]->integrate(it.second);
	}
	activate();
}

/**
 * \brief Resets context variables of recurrent layers. Called after the end of sequence
 */
void NeuralNetwork::reset()
{
	for (auto& _layer : _layers)
	{
		_layer.second->reset();
	}
}

/**
 * \brief Returns input tensors form the input layers
 * \return vector of input tensors
 */
vector<Tensor*> NeuralNetwork::get_input() {
	vector<Tensor*> result;

	for (auto& it : _input_layer)
	{
		result.push_back(_layers[it]->get_output());
	}

	return vector<Tensor*>(result);
}

/**
 * \brief Return dimension of the output layer
 * \return number of output neurons
 */
int NeuralNetwork::get_output_dim()
{
	return _layers[_output_layer]->get_dim();
}

/**
 * \brief WRONG implementation
 * \return 
 */
int NeuralNetwork::get_input_dim()
{
	int dim = 0;
	for (auto& it : _input_layer)
	{
		dim += _layers[it]->get_in_dim();
	}

	return dim;
}

/**
 * \brief Internal function which activates the rest of the network after activation of the input layers
 */
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

/**
 * \brief Adds layer to the network
 * \param p_layer layer class
 * \return pointer to the added layer (the same is on the input)
 */
BaseLayer* NeuralNetwork::add_layer(BaseLayer* p_layer) {
	_layers[p_layer->get_id()] = p_layer;

	return p_layer;
}

/**
 * \brief Returns pointer to the layer
 * \param p_layer name of the layer
 * \return pointer to layer class
 */
BaseLayer* NeuralNetwork::get_layer(const string& p_layer) {
	return _layers[p_layer];
}

/**
 * \brief Adds directed edge to the network graph connecting two layers
 * \param p_input_layer layer where connection begins
 * \param p_output_layer layer where connection ends
 */
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

/**
 * \brief Return pointer to the output tensor of the output layer
 * \return tensor pointer
 */
Tensor* NeuralNetwork::get_output()
{
	return _layers[_output_layer]->get_output();
}

/**
 * \brief Creates forward graph using breadth search tree method on network graph. Backward graph is then created by reversing forward graph.
 */
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

/**
 * \brief Serialize structure and data into json form, which can be saved to file
 * \return json class
 */
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


