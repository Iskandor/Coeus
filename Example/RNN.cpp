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
#include "ExponentialCost.h"
#include "KLDivergence.h"
#include "PowerSign.h"
#include "PackDataset2.h"
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
	//network.add_connection("hidden0", "hidden1");
	network.add_connection("hidden0", "output");
	network.init();

	BackProp algorithm(&network);

	algorithm.init(new QuadraticCost(), 0.1f, 0.9f, true);
	//Nadam algorithm(&network);
	//algorithm.init(new QuadraticCost(), 0.01f);
	algorithm.set_recurrent_mode(RTRL);

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
	/*
	IOUtils::save_network(network, "test.net");
	NeuralNetwork test(IOUtils::load_network("test.net"));

	test_add_problem(test);
	*/
}

void RNN::run_sin_prediction()
{
	vector<Tensor*> input;
	vector<Tensor*> target;

	for(int i = 0; i < 359; i++)
	{
		Tensor* ti = new Tensor({ 1 }, Tensor::ZERO);
		ti->set(0, sin(i * (2 * PI / 360)));

		Tensor* tt = new Tensor({ 1 }, Tensor::ZERO);
		tt->set(0, sin((i + 10) * (2 * PI / 360)));

		input.push_back(ti);
		target.push_back(tt);
	}

	NeuralNetwork network;
	network.add_layer(new RecurrentLayer("hidden0", 4, SIGMOID, new TensorInitializer(UNIFORM, -1e-3, 1e-3), 1));
	network.add_layer(new RecurrentLayer("hidden1", 2, SIGMOID, new TensorInitializer(UNIFORM, -1e-3, 1e-3)));
	network.add_layer(new CoreLayer("output", 1, TANH, new TensorInitializer(UNIFORM, -1e-3, 1e-3)));
	network.add_connection("hidden0", "hidden1");
	network.add_connection("hidden1", "output");
	network.init();	

	Nadam algorithm(&network);
	algorithm.init(new QuadraticCost(), 1e-3f);
	//algorithm.set_recurrent_mode(RTRL);
	//BackProp algorithm(&network);
	//algorithm.init(new QuadraticCost(), 1e-4f, 0.996f, true);

	float error = 1;

	while(error > 1e-2)
	{
		algorithm.reset();
		error = 0;
		
		for(int i = 0; i < input.size(); i++)
		{
			error += algorithm.train(input[i], target[i]);
		}

		cout << error << endl;
	}

	Logger::instance().init("sin.csv");
	network.reset();
	for (int i = 0; i < input.size(); i++)
	{
		network.activate(input[i]);
	}

	for (int i = 0; i < input.size(); i++)
	{
		network.activate(input[i]);
		cout << *network.get_output() << " " << *target[i] << endl;
		Logger::instance().log(to_string(network.get_output()->at(0)) + ";" + to_string(target[i]->at(0)));
	}
	
	/*
	float x = input[0]->at(0);

	Tensor ti({ 1 }, Tensor::ZERO);

	for (int i = 0; i < 1000; i++)
	{
		ti.set(0, x);
		network.activate(&ti);
		x = network.get_output()->at(0);
		Logger::instance().log(to_string(network.get_output()->at(0)));
	}
	*/
	Logger::instance().close();
}

void RNN::run_add_problem_gru()
{
	AddProblemDataset dataset;
	dataset.load_data("./data/add_problem_easy.dat");
	dataset.split(64);

	NeuralNetwork network;
	//network.add_layer(new GRULayer("hidden0", 4, TANH, new TensorInitializer(UNIFORM, -1e-3, 1e-3), 2));
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
	const int bound = size * 0.99;

	cout << "Size " << size << endl;

	while (correct < bound) {
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
	PackDataset2 dataset;
	dataset.load_data("./data/" + config["dataset"].get<string>() + ".csv", false, false);
	dataset.split(config["batch"].get<int>());
	//PackDataset validset;
	//validset.load_data("./data/pack_data_test.csv", false, true);


	NeuralNetwork* network;

	if (config["network"].get<string>().empty())
	{
		network = new NeuralNetwork();
		network->add_layer(new LSTMLayer("hidden0", config["hidden"].get<int>(), TANH, new TensorInitializer(LECUN_UNIFORM), dataset.get_input_dim()));
		network->add_layer(new LSTMLayer("hidden1", config["hidden"].get<int>() / 4, TANH, new TensorInitializer(LECUN_UNIFORM)));
		//network->add_layer(new LSTMLayer("hidden2", config["hidden"].get<int>() / 8, TANH, new TensorInitializer(LECUN_UNIFORM)));
		network->add_layer(new CoreLayer("output", 1, SIGMOID, new TensorInitializer(LECUN_UNIFORM)));
		network->add_connection("hidden0", "hidden1");
		//network->add_connection("hidden1", "hidden2");
		network->add_connection("hidden1", "output");
		network->init();
	}
	else
	{
		network = new NeuralNetwork(IOUtils::load_network(config["network"].get<string>()));
	}

	Nadam algorithm(network);
	algorithm.init(new QuadraticCost(), config["alpha"].get<float>());
	//BackProp algorithm(&network);
	//algorithm.init(new QuadraticCost(), config["alpha"].get<float>(), 0.9, true);
	//PowerSign algorithm(&network);
	//algorithm.init(new QuadraticCost());

	int epoch = 0;
	float last_error = -1;

	float tp = 0;
	float fp = 0;
	float tn = 0;
	float fn = 0;
	float precision = 0;
	float accuracy = 0;
	float fn_ratio = 1;
	float mcc = 0;
	const float bound = 0.999;
	const int size = dataset.data()->size();

	cout << size << " sequences" << endl;
	cout << "Training..." << endl;

	while (mcc < 1) {
		//pair<vector<Tensor*>, vector<Tensor*>> data = dataset.to_vector();
		vector<PackDataSequence2>* train = dataset.permute(true);
		vector<PackDataSequence2>* test = dataset.data();

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
				network->activate(&sequence.input);
				const int prediction = network->get_output()->at(0) > 0.5 ? 1 : 0;

				if ((*sequence.target)[0] == 1 && prediction == 1) tp++;
				if ((*sequence.target)[0] == 1 && prediction == 0) fn++;
				if ((*sequence.target)[0] == 0 && prediction == 1) fp++;
				if ((*sequence.target)[0] == 0 && prediction == 0) tn++;
			}

			precision = (fp + tp) == 0 ? 0 : (float)tp / (fp + tp);
			accuracy = (float)(tp + tn) / dataset.data()->size();
			fn_ratio = (fn + tp) == 0 ? 1 : (float)fn / (fn + tp);

			tp /= 1e4;
			tn /= 1e4;
			fp /= 1e4;
			fn /= 1e4;

			float nom = (tp * tn - fp * fn);
			float den = (tp + fp)*(tp + fn)*(tn + fp)*(tn + fn);

			if (den == 0) den = 1;

			mcc = nom / sqrt(den);

			tp *= 1e4;
			tn *= 1e4;
			fp *= 1e4;
			fn *= 1e4;

			cout << tn << " , " << fp << " , " << fn << " , " << tp << " , MCC: " << mcc << " , precision: " << precision << " , accuracy: " << accuracy << " , fn_ratio: " << fn_ratio << endl;

			if (last_error == -1 || error < last_error)
			{
				IOUtils::save_network(*network, "predictor.net");
				last_error = error;
			}
		}

		Logger::instance().log(to_string(error) + " " + to_string(mcc));

		cout << error << endl;
		cout << "Time: " << (end - start).count() * ((float)chrono::high_resolution_clock::period::num / chrono::high_resolution_clock::period::den) << endl;
		epoch++;
	}

	Logger::instance().log("Finish");
	Logger::instance().close();

	IOUtils::save_network(*network, "predictor.net");

	delete network;
}

void RNN::run_pack2()
{
	Logger::instance().init("log.log");
	Logger::instance().log("Start");
	cout << "Loading config..." << endl;
	json config = load_config("config.json");
	cout << config << endl;

	cout << "Loading dataset..." << endl;
	PackDataset2 dataset;
	dataset.load_data("./data/" + config["dataset"].get<string>() + ".csv", false, false);
	//dataset.split(config["batch"].get<int>());
	//PackDataset validset;
	//validset.load_data("./data/pack_data_test.csv", false, true);


	NeuralNetwork* network;

	if (config["network"].get<string>().empty())
	{
		network = new NeuralNetwork();
		network->add_layer(new LSTMLayer("hidden0", config["hidden"].get<int>(), TANH, new TensorInitializer(LECUN_UNIFORM), dataset.get_input_dim()));
		network->add_layer(new LSTMLayer("hidden1", config["hidden"].get<int>() / 4, TANH, new TensorInitializer(LECUN_UNIFORM)));
		network->add_layer(new LSTMLayer("hidden2", config["hidden"].get<int>() / 8, TANH, new TensorInitializer(LECUN_UNIFORM)));
		network->add_layer(new CoreLayer("output", 1, SIGMOID, new TensorInitializer(LECUN_UNIFORM)));
		network->add_connection("hidden0", "hidden1");
		network->add_connection("hidden1", "hidden2");
		network->add_connection("hidden2", "output");
		network->init();
	}
	else
	{
		network = new NeuralNetwork(IOUtils::load_network(config["network"].get<string>()));
	}

	Nadam algorithm(network);
	algorithm.init(new QuadraticCost(), config["alpha"].get<float>());
	//BackProp algorithm(&network);
	//algorithm.init(new QuadraticCost(), config["alpha"].get<float>(), 0.9, true);
	//PowerSign algorithm(&network);
	//algorithm.init(new QuadraticCost());

	int epoch = 0;
	float last_error = -1;

	float tp = 0;
	float fp = 0;
	float tn = 0;
	float fn = 0;
	float precision = 0;
	float accuracy = 0;
	float fn_ratio = 1;
	float mcc = 0;
	const float bound = 0.999;
	const int size = dataset.permute()->size();
	int _batch_index = 0;

	cout << size << " sequences" << endl;
	cout << "Training..." << endl;

	while (mcc < 1) {
		//pair<vector<Tensor*>, vector<Tensor*>> data = dataset.to_vector();
		vector<PackDataSequence3>* train = dataset.permute();
		vector<PackDataSequence3>* test = train;

		auto start = chrono::high_resolution_clock::now();
		//const float error = algorithm.train(&data.first, &data.second, config["batch"].get<int>());
		float error = 0;		

		for (auto sequence : *train)
		{			
			error += algorithm.train(&sequence.input, &sequence.target, _batch_index == 0);
			_batch_index++;
			if (_batch_index == config["batch"].get<int>()) _batch_index = 0;
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
				for (int i = 0; i < sequence.input.size(); i++)
				{
					network->activate(sequence.input[i]);

					if (sequence.target[i] != nullptr)
					{
						const int prediction = network->get_output()->at(0) > 0.5 ? 1 : 0;
						const float target = sequence.target[i]->at(0);

						if (target == 1 && prediction == 1) tp++;
						if (target == 1 && prediction == 0) fn++;
						if (target == 0 && prediction == 1) fp++;
						if (target == 0 && prediction == 0) tn++;
					}
				}
			}

			precision = (fp + tp) == 0 ? 0 : (float)tp / (fp + tp);
			accuracy = (float)(tp + tn) / dataset.data()->size();
			fn_ratio = (fn + tp) == 0 ? 1 : (float)fn / (fn + tp);

			tp /= 1e4;
			tn /= 1e4;
			fp /= 1e4;
			fn /= 1e4;

			float nom = (tp * tn - fp * fn);
			float den = (tp + fp)*(tp + fn)*(tn + fp)*(tn + fn);

			if (den == 0) den = 1;

			mcc = nom / sqrt(den);

			tp *= 1e4;
			tn *= 1e4;
			fp *= 1e4;
			fn *= 1e4;

			cout << tn << " , " << fp << " , " << fn << " , " << tp << " , MCC: " << mcc << " , precision: " << precision << " , accuracy: " << accuracy << " , fn_ratio: " << fn_ratio << endl;

			if (last_error == -1 || error < last_error)
			{
				IOUtils::save_network(*network, "predictor.net");
				last_error = error;
			}
		}

		Logger::instance().log(to_string(error) + " " + to_string(mcc));

		cout << error << endl;
		cout << "Time: " << (end - start).count() * ((float)chrono::high_resolution_clock::period::num / chrono::high_resolution_clock::period::den) << endl;
		epoch++;
	}

	Logger::instance().log("Finish");
	Logger::instance().close();

	IOUtils::save_network(*network, "predictor.net");

	delete network;
}

void RNN::test_pack_cm() const
{
	cout << "Loading dataset..." << endl;
	PackDataset2 dataset;
	dataset.load_data("./data/pack_data_test.csv", false, true);
	//dataset.load_data("./data/pack_data_train4500.csv", false, false);

	cout << "Loading network..." << endl;
	NeuralNetwork network(IOUtils::load_network("predictor.net"));

	cout << "Testing..." << endl;
	vector<PackDataSequence2>* test = dataset.data();

	int tp = 0;
	int fp = 0;
	int tn = 0;
	int fn = 0;

	for (auto sequence : *test)
	{
		network.activate(&sequence.input);
		const float output = network.get_output()->at(0);
		const int prediction = output > 0.5 ? 1 : 0;

		if ((*sequence.target)[0] == 1 && prediction == 1) {
			//cout << "TP " << network.get_output()->at(0) << endl;
			tp++;
		}
		if ((*sequence.target)[0] == 1 && prediction == 0) {
			//cout << "FN " << output << endl;
			fn++;
		}
		if ((*sequence.target)[0] == 0 && prediction == 1) {
			//cout << "FP " << output << endl;
			fp++;
		}
		if ((*sequence.target)[0] == 0 && prediction == 0) {
			tn++;
		}
	}

	cout << tn << " , " << fp << " , " << fn << " , " << tp << " , " << endl;

	float den = 1;

	if ((tp + fp) != 0 && (tp + fn) != 0 && (tn + fp) != 0 && (tn + fn) != 0)
	{
		den = sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn));
	}

	const float mcc = (tp * tn - fp * fn) / den;

	cout << mcc << endl;
}

void RNN::test_pack_alt() const
{
	cout << "Loading dataset..." << endl;
	PackDataset2 dataset;
	dataset.load_data("./data/pack_data_test.csv", false, true);

	cout << "Loading network..." << endl;
	NeuralNetwork network(IOUtils::load_network("predictor.net"));

	cout << "Testing..." << endl;
	vector<PackDataSequence2>* test = dataset.data();

	int tp = 0;
	int fp = 0;
	int tn = 0;
	int fn = 0;

	for (auto sequence : *test)
	{
		network.activate(&sequence.input);
		const int prediction = network.get_output()->at(0) > 0.5 ? 1 : 0;

		if ((*sequence.target)[0] == 1 && prediction == 1) {
			tp++;
		}
		if ((*sequence.target)[0] == 1 && prediction == 0) {
			vector<PackDataSequence2> d = dataset.create_sequence_test(sequence.player_id);

			for (int i = 0; i < d.size(); i++)
			{
				network.activate(&d[i].input);
				const int response = network.get_output()->at(0) > 0.5 ? 1 : 0;

				cout << i << " " << response << " " << network.get_output()->at(0) << endl;
			}

			fn++;
		}
		if ((*sequence.target)[0] == 0 && prediction == 1) fp++;
		if ((*sequence.target)[0] == 0 && prediction == 0) tn++;
	}

	cout << tn << " , " << fp << " , " << fn << " , " << tp << " , " << endl;
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