#include "FFN.h"
#include "NeuralNetwork.h"
#include "InputLayer.h"
#include "CoreLayer.h"
#include "BaseGradientAlgorithm.h"
#include "QuadraticCost.h"
#include "BackProph.h"
#include "RMSProp.h"
#include "Adagrad.h"
#include "Adadelta.h"
#include "ADAM.h"
#include "AdaMax.h"
#include "Nadam.h"
#include "AMSGrad.h"
#include "LSTMLayer.h"
#include "CrossEntropyCost.h"
#include "IOUtils.h"

FFN::FFN()
{
}


FFN::~FFN()
{
}

void FFN::run() {
	_network.add_layer(new InputLayer("input", 2));
	_network.add_layer(new CoreLayer("hidden", 8, SIGMOID));
	_network.add_layer(new CoreLayer("output", 1, SIGMOID));

	_network.add_connection("input", "hidden", Connection::LECUN_UNIFORM);
	_network.add_connection("hidden", "output", Connection::LECUN_UNIFORM);
	_network.init();


	double data_i[8]{ 0,0,0,1,1,0,1,1 };
	double data_t[4]{ 0,1,1,0 };

	vector<Tensor*> input;
	vector<Tensor*> target;

	for (int i = 0; i < 4; i++) {
		double *d = Tensor::alloc_arr(2);

		d[0] = data_i[i * 2];
		d[1] = data_i[i * 2 + 1];

		double *t = Tensor::alloc_arr(1);
		t[0] = data_t[i];

		input.push_back(new Tensor({ 2 }, d));
		target.push_back(new Tensor({ 1 }, t));
	}

	//BackProp model(&_network);
	//RMSProp model(&_network);
	//AdaMax model(&_network);
	//ADAM model(&_network);
	//AMSGrad model(&_network);
	Nadam model(&_network);

	//model.init(new QuadraticCost(), 0.1, 0.9, true);
	model.init(new QuadraticCost(), 0.1);

	for(int t = 0; t < 500; t++) {
		//const double error = model.train(&input, &target);
		double error = 0;

		for(int i = 0; i < 4; i++)
		{
			error += model.train(input[i], target[i]);
		}

		cout << "Error: " << error << endl;
	}

	cout << endl;

	for (int i = 0; i < 4; i++) {
		_network.activate(input[i]);
		cout << _network.get_output()->at(0) << endl;
	}

	/*
	NeuralNetwork copy(_network);
	copy.init();

	cout << endl;
	for (int i = 0; i < 4; i++) {
		copy.activate(input[i]);
		cout << copy.get_output()->at(0) << endl;
	}

	IOUtils::save_network(copy, "test.net");

	NeuralNetwork loaded(IOUtils::load_network("test.net"));

	cout << endl;
	for (int i = 0; i < 4; i++) {
		loaded.activate(input[i]);
		cout << loaded.get_output()->at(0) << endl;
	}
	*/

	for (int i = 0; i < 4; i++) {
		delete input[i];
		delete target[i];
	}

	
}

void FFN::run_iris() {
	_dataset.load_data("./data/iris.data");

	_network.add_layer(new InputLayer("input", IrisDataset::SIZE));
	_network.add_layer(new CoreLayer("hidden", 256, SIGMOID));
	_network.add_layer(new CoreLayer("output", 3, SOFTMAX));

	_network.add_connection("input", "hidden", Connection::LECUN_UNIFORM);
	_network.add_connection("hidden", "output", Connection::LECUN_UNIFORM);
	_network.init();


	const int epochs = 500;
	vector<IrisDatasetItem>* data = nullptr;
	map<int, Tensor> target;

	for(int i = 0; i < IrisDataset::CATEGORIES; i++) {
		target[i] = Tensor::Zero({ IrisDataset::CATEGORIES });
		target[i][i] = 1;
	}

	//BackProp model(&_network);
	//model.init(new CrossEntropyCost(), 0.0001, 0.9, false);
	ADAM model(&_network);
	model.init(new CrossEntropyCost(), 0.0002);

	for (int t = 0; t < epochs; t++) {
		data = _dataset.permute();
		double error = 0;

		for (int i = 0; i < data->size(); i++) {
			error += model.train(data->at(i).data, &target[(*_dataset.get_target_map())[data->at(i).target]]);						
		}
		cout << "Error: " << error << endl;
	}

	for (int i = 0; i < data->size(); i++) {
		_network.activate(data->at(i).data);
		for (int o = 0; o < 3; o++) {
			cout << _network.get_output()->at(o) << " , ";
		}
		cout << data->at(i).target << endl;

	}

}
