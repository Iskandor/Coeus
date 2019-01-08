#include "RNN.h"
#include "InputLayer.h"
#include "LSTMLayer.h"
#include "CoreLayer.h"
#include "QuadraticCost.h"
#include "BPTT.h"
#include "AddProblemDataset.h"
#include "IOUtils.h"
#include "RMSProp.h"
#include "KLDivergence.h"
#include "ExponentialCost.h"
#include "HellingerDistance.h"
#include "PackDataset.h"
#include "Nadam.h"
#include "BackProph.h"
#include "ADAM.h"
#include <chrono>
#include "Logger.h"
#include "PowerSign.h"


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
	RMSProp model(&network);
	//AdaMax model(&_network);
	//ADAM algorithm(&network);
	//AMSGrad model(&_network);
	//Nadam model(&_network);

	//model.init(new QuadraticCost(), 0.1, 0.99, true);
	model.init(new QuadraticCost(), 0.05);

	for (int t = 0; t < 2000; t++) {
		double error = 0;
		for (int i = 0; i < 4; i++) {
			error += model.train(&input[i], &target[i]);
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

	network.add_connection("input", "hidden", Connection::LECUN_UNIFORM);
	network.add_connection("hidden", "output", Connection::LECUN_UNIFORM);

	network.init();

	//ADAM algorithm(&network);
	Nadam algorithm(&network);
	algorithm.init(new QuadraticCost(), 1e-3);
	//algorithm.init(new QuadraticCost(), 0.0);
	//BackProp algorithm(&network);
	//algorithm.init(new QuadraticCost(), 0.01, 0.9, true);
	//PowerSign algorithm(&network);
	//algorithm.init(new QuadraticCost());

	int epochs = 0;
	int correct = 0;
	const int size = dataset.permute()->size();
	const int bound =  size * 0.99;

	while(correct < bound) {
		pair<vector<Tensor*>, vector<Tensor*>> data = dataset.to_vector();
		vector<AddProblemSequence>* test = dataset.data();

		const double error = algorithm.train(&data.first, &data.second, 16);
		//double error = 0;
		correct = 0;

		for (auto sequence : *test)
		{
			//error += algorithm.train(&sequence.input, &sequence.target);

			network.activate(&sequence.input);
			//cout << *network.get_output() << endl;

			if (abs(network.get_output()->at(0) - sequence.target[0]) < 0.04)
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

void RNN::run_pack()
{
	Logger::instance().init("log.log");
	Logger::instance().log("Start");
	cout << "Loading config..." << endl;
	json config = load_config("config.json");
	cout << config << endl;

	cout << "Loading dataset..." << endl;
	PackDataset dataset;
	dataset.load_data("./data/pack_data_red.csv");

	//NeuralNetwork network(IOUtils::load_network("predictor.net"));
	NeuralNetwork network;
	network.add_layer(new InputLayer("input", 230));
	network.add_layer(new LSTMLayer("hidden", config["hidden"].get<int>(), TANH));
	network.add_layer(new CoreLayer("output", 1, SIGMOID));

	network.add_connection("input", "hidden", Connection::LECUN_UNIFORM);
	network.add_connection("hidden", "output", Connection::LECUN_UNIFORM);

	network.init();

	Nadam algorithm(&network);
	algorithm.init(new QuadraticCost(), config["alpha"].get<double>());
	//BackProp algorithm(&network);
	//algorithm.init(new QuadraticCost(), config["alpha"].get<double>(), 0.9, true);
	//PowerSign algorithm(&network);
	//algorithm.init(new QuadraticCost(), config["alpha"].get<double>());

	int epoch = 0;
	int correct = 0;
	const int size = dataset.data()->size();
	const int bound = size * 0.99;

	cout << "Training..." << endl;

	while (correct < bound) {
		pair<vector<Tensor*>, vector<Tensor*>> data = dataset.to_vector();
		vector<PackDataSequence>* test = dataset.data();

		auto start = chrono::high_resolution_clock::now();
		const double error = algorithm.train(&data.first, &data.second, config["batch"].get<int>());
		auto end = chrono::high_resolution_clock::now();

		//double error = 0;
		if (epoch % config["evaluate"].get<int>() == 0)
		{
			correct = 0;

			for (auto sequence : *test)
			{
				//error += algorithm.train(&sequence.input, &sequence.target);

				network.activate(&sequence.input);

				if (abs(network.get_output()->at(0) - sequence.target[0]) < 0.01)
				{
					correct++;
				}
			}
			cout << correct << " / " << size << endl;
		}

		Logger::instance().log(to_string(error) + " " + to_string(correct) + " " + to_string(size));

		cout << error << endl;
		cout << "Time: " << (end - start).count() * ((double)chrono::high_resolution_clock::period::num / chrono::high_resolution_clock::period::den) << endl;
		epoch++;
	}

	Logger::instance().log("Finish");
	Logger::instance().close();

	IOUtils::save_network(network, "predictor.net");
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

		if (abs(p_network.get_output()->at(0) - sequence.target[0]) < 0.04)
		{
			correct++;
		}
	}

	cout << correct << " / " << data->size() << endl;
}

void RNN::test_pack() const
{
	cout << "Loading dataset..." << endl;
	PackDataset dataset;
	dataset.load_data("./data/pack_data.csv");

	cout << "Loading network..." << endl;
	NeuralNetwork network(IOUtils::load_network("predictor.net"));

	cout << "Testing..." << endl;
	vector<PackDataSequence>* test = dataset.data();

	const int size = dataset.data()->size();
	int correct30 = 0;
	int correct1 = 0;
	double error = 0;
	QuadraticCost c;

	for (auto sequence : *test)
	{
		network.activate(&sequence.input);

		if (abs(network.get_output()->at(0) - sequence.target[0]) < 0.01)
		{
			correct1++;
		}
		else
		if (abs(network.get_output()->at(0) - sequence.target[0]) < 0.3)
		{
			correct30++;
		}

		error += c.cost(network.get_output(), &sequence.target);

		//cout << *network.get_output() << " - " << sequence.target[0] << endl;
	}

	cout << correct1 << " / " << correct1 + correct30 << " / " << size << endl;
	cout << error << endl;
}

json RNN::load_config(const string& p_filename) const
{
	json data;
	ifstream file;

	try {
		file.open(p_filename);
	}
	catch (std::ios_base::failure& e) {
		std::cerr << e.what() << '\n';
	}

	if (file.is_open()) {
		file >> data;
	}
	file.close();

	return json(data);
}
