#include "IrisTest.h"
#include "LSOM_params.h"
#include "LSOM_learning.h"
#include "SOM_learning.h"
#include <fstream>


IrisTest::IrisTest() {
	_lsom = nullptr;
	_dataset.load_data("./data/iris.data");
	_dataset.encode();
}

IrisTest::~IrisTest()
{
	delete _lsom;
}

void IrisTest::init() {
	_lsom = new LSOM("LSOM", 128, 5, 5, TANH);
	//_lsom = new SOM("LSOM", 4, 8, 8, NeuralGroup::EXPONENTIAL);
}

void IrisTest::run(const int p_epochs) {
	SOM_analyzer analyzer;
	//SOM_params params(_lsom);
	//params.init_training(0.8, p_epochs);
	LSOM_params params(_lsom);
	params.init_training(1e-4, 1e-4, p_epochs);
	LSOM_learning learner(_lsom, &params, &analyzer);
	//SOM_learning learner(_lsom, &params, &analyzer);

	vector<IrisDatasetItem>* data = nullptr;

	for(int t = 0; t < p_epochs; t++) {
		data = _dataset.permute();

		const auto start = chrono::system_clock::now();

		for(int i = 0; i < data->size(); i++) {
			//cout << data->at(i).target << endl;
			learner.train(data->at(i).data);
		}

		const auto end = chrono::system_clock::now();
		chrono::duration<double> elapsed_seconds = end - start;
		if (t % 10 == 0) {
			cout << "Epoch " << t << endl;
			cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
			cout << " LSOM qError: " << analyzer.q_error() << " WD: " << analyzer.winner_diff(_lsom->get_lattice()->get_dim()) << endl;

			for (int i = 0; i < _lsom->get_lattice()->get_dim(); i++) {
				double sum = 0;
				for (int j = 0; j < _lsom->get_lattice()->get_dim(); j++) {
					sum += _lsom->get_lateral()->get_weights()->at(i, j);
				}
				cout << i << " " << sum << endl;
			}
		}

		//learner.update_friendship();
		learner.update();
		analyzer.end_epoch();
		params.param_decay();		
	}

	/*
	for(int i = 0; i < learner._friendship.size(); i++) {
		cout << i << " " << learner._friendship[i] << endl;
	}
	cout << endl;
	*/
	for (int i = 0; i < _lsom->get_lattice()->get_dim(); i++) {
		double sum = 0;
		for (int j = 0; j < _lsom->get_lattice()->get_dim(); j++) {
			sum += _lsom->get_lateral()->get_weights()->at(i, j);
		}
		cout << i << " " << sum << endl;
	}
}

void IrisTest::test() {
	double* activity = new double[_lsom->dim_x() * _lsom->dim_y() * IrisDataset::CATEGORIES]{ 0 };

	vector<IrisDatasetItem>* data = _dataset.permute();

	for (int i = 0; i < data->size(); i++) {
		//cout << data->at(i).target << endl;
		_lsom->activate(data->at(i).data);
		for (int n = 0; n < _lsom->get_lattice()->get_dim(); n++) {
			activity[n * IrisDataset::CATEGORIES + (*_dataset.get_target_map())[data->at(i).target]] += max(0.0, _lsom->get_output()->at(n));
		}
	}

	save_results("iris_test.act", _lsom->dim_x(), _lsom->dim_y(), activity, IrisDataset::CATEGORIES);

	delete[] activity;
}

void IrisTest::save_results(const string p_filename, const int p_dim_x, const int p_dim_y, double* p_data, 	const int p_category) const {
	ofstream file(p_filename);

	if (file.is_open()) {
		file << p_dim_x << "," << p_dim_y << endl;
		for (int i = 0; i < p_dim_x * p_dim_y; i++) {
			for (int j = 0; j < p_category; j++) {
				if (j == p_category - 1) {
					file << p_data[i * p_category + j];
				}
				else {
					file << p_data[i * p_category + j] << ",";
				}
			}
			if (i < p_dim_x * p_dim_y - 1) file << endl;
		}
	}

	file.close();
}


