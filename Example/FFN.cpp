#include "FFN.h"
#include "NeuralNetwork.h"
#include "CoreLayer.h"
#include "GradientAlgorithm.h"
#include "QuadraticCost.h"
#include "RMSProp.h"
#include "LSTMLayer.h"
#include "CrossEntropyCost.h"
#include "IOUtils.h"
#include <chrono>
#include "Adadelta.h"
#include "Adagrad.h"
#include "ADAM.h"
#include "AdaMax.h"
#include "AMSGrad.h"
#include "BackProph.h"
#include "Nadam.h"
#include "PowerSign.h"
#include "RecurrentLayer.h"

FFN::FFN()
{
}


FFN::~FFN()
{
}

void FFN::run() {
	float data_i[8]{ 0,0,0,1,1,0,1,1 };
	float data_t[4]{ 0,1,1,0 };

	vector<Tensor*> o_input;
	vector<Tensor*> o_target;

	for (int i = 0; i < 4; i++) {
		float *d = Tensor::alloc_arr(2);

		d[0] = data_i[i * 2];
		d[1] = data_i[i * 2 + 1];

		float *t = Tensor::alloc_arr(1);
		t[0] = data_t[i];

		o_input.push_back(new Tensor({ 2 }, d));
		o_target.push_back(new Tensor({ 1 }, t));
	}

	Tensor input({ 4, 2 }, Tensor::ZERO);
	Tensor target({ 4, 1 }, Tensor::ZERO);

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 2; j++)
		{
			input.set(i, j, data_i[i * 2 + j]);
		}

		target.set(i, 0, data_t[i]);
	}

	NeuralNetwork network;

	network.add_layer(new CoreLayer("hidden", 4, SIGMOID, new TensorInitializer(LECUN_UNIFORM), 2));
	network.add_layer(new CoreLayer("output", 1, SIGMOID, new TensorInitializer(LECUN_UNIFORM)));
	network.add_connection("hidden", "output");

	network.init();

	BackProp optimizer(&network);

	optimizer.init(new QuadraticCost(), 0.5f, 0.9f, true);

	const auto start = chrono::system_clock::now();

	for (int t = 0; t < 1000; t++) {
		float error = 0;

		/*
		for (int i = 0; i < 4; i++)
		{
			error += optimizer.train(o_input[i], o_target[i]);
		}
		*/

		error = optimizer.train(&input, &target);
		cout << error << endl;
	}

	const auto end = chrono::system_clock::now();
	chrono::duration<float> elapsed_seconds = end - start;
	cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

	for (int i = 0; i < 4; i++) {
		network.activate(o_input[i]);
		cout << *network.get_output() << endl;
	}
}

void FFN::run_iris() {
	_dataset.load_data("./data/iris.data");

	NeuralNetwork network;
	network.add_layer(new CoreLayer("hidden", 16, SIGMOID, new TensorInitializer(LECUN_UNIFORM), IrisDataset::SIZE));
	network.add_layer(new CoreLayer("output", IrisDataset::CATEGORIES, SOFTMAX, new TensorInitializer(LECUN_UNIFORM)));
	network.add_connection("hidden", "output");
	network.init();


	const int epochs = 1000;
	vector<IrisDatasetItem>* data = nullptr;
	map<int, Tensor> target;

	for(int i = 0; i < IrisDataset::CATEGORIES; i++) {
		target[i] = Tensor::Zero({ IrisDataset::CATEGORIES });
		target[i][i] = 1;
	}

	RMSProp model(&network);
	//model.init(new CrossEntropyCost(), 0.001f);
	model.init(new QuadraticCost(), 0.001f);

	for (int t = 0; t < epochs; t++) {
		data = _dataset.permute();
		float error = 0;

		for (int i = 0; i < data->size(); i++) {
			error += model.train(data->at(i).data, &target[(*_dataset.get_target_map())[data->at(i).target]]);						
		}
		cout << "Error: " << error << endl;
	}

	for (int i = 0; i < data->size(); i++) {
		network.activate(data->at(i).data);
		for (int o = 0; o < 3; o++) {
			cout << network.get_output()->at(o) << " , ";
		}
		cout << data->at(i).target << endl;

	}
}
