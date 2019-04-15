#include "RNN.h"
#include "LSTMLayer.h"
#include "CoreLayer.h"
#include "QuadraticCost.h"
#include "AddProblemDataset.h"
#include "IOUtils.h"
#include "BackProph.h"

#include <chrono>
#include "RecurrentLayer.h"
#include "ADAM.h"


RNN::RNN()
{
}


RNN::~RNN()
{
}

void RNN::run_add_problem()
{
	AddProblemDataset dataset;
	dataset.load_data("./data/add_problem_easy.dat");
	dataset.split(64);

	NeuralNetwork network;
	//network.add_layer(new RecurrentLayer("hidden0", 4, TANH, new TensorInitializer(UNIFORM, -1e-3, 1e-3), 2));
	network.add_layer(new LSTMLayer("hidden0", 4, TANH, new TensorInitializer(UNIFORM, -1e-3, 1e-3), 2));
	network.add_layer(new CoreLayer("output", 1, SIGMOID, new TensorInitializer(UNIFORM, -1e-3, 1e-3)));
	network.add_connection("hidden0", "output");
	network.init();

	BackProp algorithm(&network);
	algorithm.init(new QuadraticCost(), 0.1f, 0.9f, true);
	//ADAM algorithm(&network);
	//algorithm.init(new QuadraticCost(), 0.01f);

	int epochs = 0;
	int correct = 0;
	const int size = dataset.raw_data()->size();
	const int bound =  size * 0.99;

	cout << "Size " << size << endl;

	while(correct < bound) {
		//pair<vector<vector<Tensor*>>, vector<Tensor*>> data = dataset.to_vector();
		vector<AddProblemSequence>* train_b = dataset.permute(true);
		//vector<AddProblemSequence>* train_s = dataset.permute(false);
		vector<AddProblemSequence>* test = dataset.raw_data();

		float error = 0;
		correct = 0;

		for (auto sequence : *train_b)
		{
			error += algorithm.train(&sequence.input, sequence.target);
		}

		/*
		for (auto sequence : *train_s)
		{
			error += algorithm.train(&sequence.input, sequence.target);
		}
		*/

		for (auto sequence : *test)
		{
			//error += algorithm.train(&sequence.input, sequence.target);
			network.activate(&sequence.input);

			if (abs(network.get_output()->at(0) - (*sequence.target)[0]) < 0.04)
			{
				correct++;
			}
		}

		epochs++;
		cout << error << endl;
		cout << correct << " / " << size << endl;
	}
	cout << epochs << endl;

	test_add_problem(network);
}

void RNN::test_add_problem(NeuralNetwork& p_network) const
{
	AddProblemDataset testset;
	testset.load_data("./data/add_problem_easy_test.dat");

	vector<AddProblemSequence>* data = testset.data();

	int correct = 0;

	for (auto sequence : *data)
	{
		p_network.activate(&sequence.input);
		cout << p_network.get_output()->at(0) << " - " << sequence.target[0] << endl;

		if (abs(p_network.get_output()->at(0) - (*sequence.target)[0]) < 0.04)
		{
			correct++;
		}
	}

	cout << correct << " / " << data->size() << endl;
}