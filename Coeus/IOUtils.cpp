#include "IOUtils.h"
#include <fstream>
#include "LinearActivation.h"
#include "BinaryActivation.h"
#include "SigmoidActivation.h"
#include "TanhActivation.h"
#include "SoftplusActivation.h"
#include "ReluActivation.h"
#include "SoftmaxActivation.h"
#include "RecurrentLayer.h"
#include "CoreLayer.h"
#include "LSTMLayer.h"

using namespace Coeus;

IOUtils::IOUtils()
= default;

IOUtils::~IOUtils()
= default;

void IOUtils::save_network(NeuralNetwork& p_network, const string& p_filename)
{
	json data;

	data["header"] = VERSION;
	data["network"] = p_network.get_json();

	ofstream file;
	file.open(p_filename);
	file << data.dump();
	file.close();
}

json IOUtils::load_network(const string& p_filename)
{
	json data;
	ifstream file;

	file.open(p_filename);
	file >> data;
	file.close();

	string header = data["header"];

	return data["network"];
}

BaseLayer* IOUtils::create_layer(const json& p_data)
{
	BaseLayer* layer = nullptr;

	const auto type = static_cast<BaseLayer::TYPE>(p_data["type"].get<int>());

	switch(type)
	{
	case BaseLayer::SOM: 
		break;
	case BaseLayer::MSOM: 
		break;
	case BaseLayer::CORE:
		layer = create_layer<CoreLayer>(p_data);
		break;
	case BaseLayer::RECURRENT:
		layer = create_layer<RecurrentLayer>(p_data);
		break;
	case BaseLayer::LSTM:
		layer = create_layer<LSTMLayer>(p_data);
		break;
	case BaseLayer::LSOM:
		break;
	default: 
		layer = nullptr;
	}

	return layer;
}

IActivationFunction* IOUtils::init_activation_function(json p_data)
{
	IActivationFunction* f;

	const auto type = static_cast<ACTIVATION>(p_data["type"].get<int>());

	switch (type) {
	case LINEAR:
		f = new LinearActivation();
		break;
	case BINARY:
		f = new BinaryActivation();
		break;
	case SIGMOID:
		f = new SigmoidActivation();
		break;
	case TANH:
		f = new TanhActivation();
		break;
	case SOFTPLUS:
		f = new SoftplusActivation();
		break;
	case RELU:
		f = new ReluActivation();
		break;
	case SOFTMAX:
		f = new SoftmaxActivation();
		break;
	default:
		f = nullptr;
	}

	return f;
}

json IOUtils::save_param(Param * p_param)
{
	json data;

	data["id"] = p_param->get_id();
	data["rank"] = p_param->get_data()->rank();

	stringstream ss_shape;

	for (int i = 0; i < p_param->get_data()->rank(); i++) {
		int dim = p_param->get_data()->shape(i);
		ss_shape.write((char*)&dim, sizeof(int));
	}

	data["shape"] = ss_shape.str();

	stringstream ss_data;

	for (int i = 0; i < p_param->get_data()->size(); i++) {
		float w = (*p_param->get_data())[i];
		ss_data.write((char*)&w, sizeof(float));
	}

	data["values"] = ss_data.str();

	return data;
}

Param* IOUtils::load_param(json p_data)
{
	int rank = p_data["rank"].get<int>();
	int* shape = Tensor::alloc_shape(rank);

	stringstream ss_shape(p_data["shape"].get<string>());
	ss_shape.seekg(0, ios::end);
	streampos size = ss_shape.tellg();
	ss_shape.seekg(0, ios::beg);
	ss_shape.read(reinterpret_cast<char*>(shape), size);

	int data_size = 1;

	for(int i = 0; i < rank; i++)
	{
		data_size *= shape[i];
	}

	float* data = Tensor::alloc_arr(data_size);

	stringstream ss_data(p_data["values"].get<string>());
	ss_data.seekg(0, ios::end);
	size = ss_data.tellg();
	ss_data.seekg(0, ios::beg);
	ss_data.read(reinterpret_cast<char*>(data), size);

	return new Param(p_data["id"].get<string>(), new Tensor(rank, shape, data));
}
