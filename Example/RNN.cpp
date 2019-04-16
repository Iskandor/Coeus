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
#include "Logger.h"
#include "PackDataset.h"
#include "Nadam.h"
#include "CrossEntropyCost.h"


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

void RNN::run_pack()
{
	Logger::instance().init("log.log");
	Logger::instance().log("Start");
	cout << "Loading config..." << endl;
	json config = load_config("config.json");
	cout << config << endl;

	cout << "Loading dataset..." << endl;
	PackDataset dataset;
	dataset.load_data("./data/" + config["dataset"].get<string>() + ".csv");
	dataset.split(config["batch"].get<int>());
	//PackDataset validset;
	//validset.load_data("./data/pack_data_test.csv", false, true);

	//NeuralNetwork network(IOUtils::load_network("predictor.net"));

	NeuralNetwork network;
	network.add_layer(new LSTMLayer("hidden0", config["hidden"].get<int>(), TANH, new TensorInitializer(LECUN_UNIFORM), 230));
	network.add_layer(new CoreLayer("output", 1, SIGMOID, new TensorInitializer(LECUN_UNIFORM)));

	network.add_connection("hidden0", "output");

	network.init();

	Nadam algorithm(&network);
	algorithm.init(new CrossEntropyCost(), config["alpha"].get<float>());
	//BackProp algorithm(&network);
	//algorithm.init(new QuadraticCost(), config["alpha"].get<float>(), 0.8, true);
	//PowerSign algorithm(&network);
	//algorithm.init(new QuadraticCost(), config["alpha"].get<float>());

	int epoch = 0;

	int tp = 0;
	int fp = 0;
	int tn = 0;
	int fn = 0;
	float precision = 0;
	float accuracy = 0;
	float fn_ratio = 1;
	const float bound = 0.999;
	const int size = dataset.data()->size();

	cout << size << " sequences" << endl;
	cout << "Training..." << endl;

	while (precision < bound || accuracy < bound || fn_ratio >(1 - bound)) {
		//pair<vector<Tensor*>, vector<Tensor*>> data = dataset.to_vector();
		vector<PackDataSequence>* train = dataset.permute(true);
		vector<PackDataSequence>* test = dataset.data();

		auto start = chrono::high_resolution_clock::now();
		//const float error = algorithm.train(&data.first, &data.second, config["batch"].get<int>());
		float error = 0;

		for (auto sequence : *train)
		{
			error += algorithm.train(&sequence.input, sequence.target);
		}

		auto end = chrono::high_resolution_clock::now();

		if (epoch % config["evaluate"].get<int>() == 0)
		{
			tp = 0;
			fp = 0;
			tn = 0;
			fn = 0;

			for (auto sequence : *test)
			{
				network.activate(&sequence.input);
				const int prediction = network.get_output()->at(0) > 0.5 ? 1 : 0;

				if ((*sequence.target)[0] == 1 && prediction == 1) tp++;
				if ((*sequence.target)[0] == 1 && prediction == 0) fn++;
				if ((*sequence.target)[0] == 0 && prediction == 1) fp++;
				if ((*sequence.target)[0] == 0 && prediction == 0) tn++;
			}

			precision = (fp + tp) == 0 ? 0 : (float)tp / (fp + tp);
			accuracy = (float)(tp + tn) / dataset.data()->size();
			fn_ratio = (fn + tp) == 0 ? 1 : (float)fn / (fn + tp);

			cout << tn << " , " << fp << " , " << fn << " , " << tp << " , " << precision << " , " << accuracy << " , " << fn_ratio << endl;
			IOUtils::save_network(network, "predictor.net");
		}

		Logger::instance().log(to_string(error) + " " + to_string(precision) + " " + to_string(accuracy) + " " + to_string(fn_ratio));

		cout << error << endl;
		cout << "Time: " << (end - start).count() * ((float)chrono::high_resolution_clock::period::num / chrono::high_resolution_clock::period::den) << endl;
		epoch++;
	}

	Logger::instance().log("Finish");
	Logger::instance().close();

	IOUtils::save_network(network, "predictor.net");
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