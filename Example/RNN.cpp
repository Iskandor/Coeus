#include "RNN.h"
#include "InputLayer.h"
#include "LSTMLayer.h"
#include "CoreLayer.h"
#include "QuadraticCost.h"
#include "ADAM.h"
#include "BPTT.h"
#include "AddProblemDataset.h"
#include "BackProph.h"


RNN::RNN()
{
}


RNN::~RNN()
{
}

void RNN::run()
{
	_network.add_layer(new InputLayer("input", 2));
	_network.add_layer(new LSTMLayer("hidden", 4, SIGMOID));
	_network.add_layer(new CoreLayer("output", 1, SIGMOID));

	_network.add_connection("input", "hidden", Connection::UNIFORM, 0.1);
	_network.add_connection("hidden", "output", Connection::UNIFORM, 0.1);

	_network.init();

	double data_i[4][2]{ {0,0},{0,1},{1,0},{1,1} };
	double data_t[4]{ 0,1,1,0 };

	Tensor input[4];
	Tensor target[4];

	for (int i = 0; i < 4; i++) {
		double *d = Tensor::alloc_arr(2);
		d[0] = data_i[i][0];
		d[1] = data_i[i][1];
		input[i] = Tensor({ 2 }, d);

		double *t = Tensor::alloc_arr(1);
		t[0] = data_t[i];
		target[i] = Tensor({ 1 }, t);
	}

	//BackProp model(&_network);
	//RMSProp model(&_network);
	//AdaMax model(&_network);
	ADAM algorithm(&_network);
	//AMSGrad model(&_network);
	//Nadam model(&_network);

	//model.init(new QuadraticCost(), 0.1, 0.99, true);
	algorithm.init(new QuadraticCost(), 0.05);

	for (int t = 0; t < 2000; t++) {
		double error = 0;
		for (int i = 0; i < 4; i++) {
			error += algorithm.train(&input[i], &target[i]);
		}
		cout << error << endl;
	}

	cout << endl;

	for (int i = 0; i < 4; i++) {
		_network.activate(&input[i]);
		cout << _network.get_output()->at(0) << endl;
	}
}

void RNN::run_add_problem()
{
	AddProblemDataset dataset;
	dataset.load_data("./data/add_problem_easy.dat");

	_network.add_layer(new InputLayer("input", 2));
	_network.add_layer(new LSTMLayer("hidden", 64, TANH));
	_network.add_layer(new CoreLayer("output", 1, SIGMOID));

	_network.add_connection("input", "hidden", Connection::UNIFORM, 0.1);
	_network.add_connection("hidden", "output", Connection::UNIFORM, 0.1);

	_network.init();

	ADAM algorithm(&_network);
	algorithm.init(new QuadraticCost(), 0.001);

	double error = 1;

	while(error > 0.01) {
		vector<AddProblemSequence>* data = dataset.permute();
		error = 0;
		_network.reset();

		for (int i = 0; i < data->size(); i++) {

			AddProblemSequence sequence = data->at(i);

			for(int s = 0; s < sequence.input.size() - 1; s++)
			{
				algorithm.train(&sequence.input[s], nullptr);
			}

			error += algorithm.train(&sequence.input[sequence.input.size() - 1], &sequence.target);
			
		}
		cout << error << endl;
	}

	vector<AddProblemSequence>* data = dataset.permute();

	for (int i = 0; i < 20; i++) {
		AddProblemSequence sequence = data->at(i);
		_network.reset();

		for (int s = 0; s < sequence.input.size() - 1; s++)
		{
			_network.activate(&sequence.input[s]);
		}
		cout << _network.get_output()->at(0) << " - " << sequence.target[0] << endl;
		
	}
}
