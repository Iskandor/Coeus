#include "RNN.h"
#include "InputLayer.h"
#include "LSTMLayer.h"
#include "CoreLayer.h"
#include "QuadraticCost.h"
#include "ADAM.h"
#include "BPTT.h"
#include "AddProblemDataset.h"
#include "BackProph.h"
#include "IOUtils.h"
#include "Nadam.h"
#include "AMSGrad.h"
#include "RMSProp.h"
#include "KLDivergence.h"
#include "ExponentialCost.h"
#include "HellingerDistance.h"


RNN::RNN()
{
}


RNN::~RNN()
{
}

void RNN::run()
{
	NeuralNetwork network;
	network.add_layer(new InputLayer("input", 2));
	network.add_layer(new LSTMLayer("hidden", 4, SIGMOID));
	network.add_layer(new CoreLayer("output", 1, SIGMOID));

	network.add_connection("input", "hidden", Connection::UNIFORM, 0.1);
	network.add_connection("hidden", "output", Connection::UNIFORM, 0.1);

	network.init();

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
	ADAM algorithm(&network);
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
		network.activate(&input[i]);
		cout << network.get_output()->at(0) << endl;
	}
}

void RNN::run_add_problem()
{
	AddProblemDataset dataset;
	dataset.load_data("./data/add_problem_easy.dat");

	NeuralNetwork network;
	network.add_layer(new InputLayer("input", 2));
	network.add_layer(new LSTMLayer("hidden", 4, TANH));
	network.add_layer(new CoreLayer("output", 1, SIGMOID));

	network.add_connection("input", "hidden", Connection::UNIFORM, 0.1);
	network.add_connection("hidden", "output", Connection::UNIFORM, 0.1);

	network.init();

	Nadam algorithm(&network);
	algorithm.init(new QuadraticCost(), 0.0001);
	//BackProp algorithm(&network);
	//algorithm.init(new QuadraticCost(), 0.1, 0.9, true);

	
	int correct = 0;
	const int bound = dataset.permute()->size() * 0.99;

	while(correct < bound) {
		vector<AddProblemSequence>* data = dataset.permute();
		double error = 0;
		correct = 0;

		for (auto sequence : *data)
		{
			network.reset();

			for(int s = 0; s < sequence.input.size() - 1; s++)
			{
				algorithm.train(&sequence.input[s], nullptr);
			}

			error += algorithm.train(&sequence.input[sequence.input.size() - 1], &sequence.target);

			if (abs(network.get_output()->at(0) - sequence.target[0]) < 0.04)
			{
				correct++;
			}
			
		}
		cout << error << endl;
		cout << correct << " / " << data->size() << endl;
	}

	IOUtils::save_network(network, "add_problem.net");
	test(network);

	NeuralNetwork network_test(IOUtils::load_network("add_problem.net"));
	test(network_test);
}

void RNN::test(NeuralNetwork& p_network) const
{
	AddProblemDataset testset;
	testset.load_data("./data/add_problem_easy_test.dat");

	vector<AddProblemSequence>* data = testset.data();

	int correct = 0;

	for (auto sequence : *data)
	{
		p_network.reset();

		for (int s = 0; s < sequence.input.size(); s++)
		{
			p_network.activate(&sequence.input[s]);
		}
		cout << p_network.get_output()->at(0) << " - " << sequence.target[0] << endl;

		if (abs(p_network.get_output()->at(0) - sequence.target[0]) < 0.04)
		{
			correct++;
		}
	}

	cout << correct << " / " << data->size() << endl;
}
