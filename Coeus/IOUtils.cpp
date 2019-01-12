#include "IOUtils.h"
#include <fstream>
#include "InputLayer.h"
#include "CoreLayer.h"
#include "LinearActivation.h"
#include "BinaryActivation.h"
#include "SigmoidActivation.h"
#include "TanhActivation.h"
#include "SoftplusActivation.h"
#include "ReluActivation.h"
#include "SoftmaxActivation.h"
#include "LSTMLayer.h"

using namespace Coeus;

IOUtils::IOUtils()
= default;

IOUtils::~IOUtils()
= default;

void IOUtils::save_network(NeuralNetwork& p_network, const string& p_filename)
{
	json data;

	data["header"] = strcat("Coeus ", VERSION);
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

BaseLayer* IOUtils::create_layer(json p_data)
{
	BaseLayer* layer = nullptr;

	const auto type = static_cast<BaseLayer::TYPE>(p_data["type"].get<int>());

	switch(type)
	{
	case BaseLayer::SOM: 
		break;
	case BaseLayer::MSOM: 
		break;
	case BaseLayer::INPUT:
		layer = create_layer<InputLayer>(p_data);
		break;
	case BaseLayer::CORE:
		layer = create_layer<CoreLayer>(p_data);
		break;
	case BaseLayer::RECURRENT: 
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
