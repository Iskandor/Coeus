//
// Created by user on 14. 12. 2017.
//

#include <fstream>
#include <chrono>
#include "ModelMNS.h"
#include "MSOM_learning.h"
#include "SOM_learning.h"
#include <iostream>
#include <string>
#include "IOUtils.h"
#include <ppl.h>
#include <concrtrm.h>
#include "Config.h"

using namespace MNS;
using namespace Concurrency;
using namespace std;

ModelMNS::ModelMNS() {
	_F5 = nullptr;
	_STS = nullptr;

}

ModelMNS::~ModelMNS() {
	delete _F5;
	delete _STS;
}

void ModelMNS::init(string p_timestamp) {
	cout << "Loading data...";
	_data.loadData(Config::instance().visual_data, Config::instance().motor_data);
	cout << "done" << endl;

	if (p_timestamp.empty()) {
		cout << "Initializing networks...";

		_F5 = new MSOM(
			_sizeF5input,
			Config::instance().f5_config.dim_x,
			Config::instance().f5_config.dim_y,
			NeuralGroup::EXPONENTIAL,
			Config::instance().f5_config.alpha,
			Config::instance().f5_config.beta);

		_STS = new MSOM(_sizeSTSinput,
			Config::instance().sts_config.dim_x,
			Config::instance().sts_config.dim_y,
			NeuralGroup::EXPONENTIAL,
			Config::instance().sts_config.alpha,
			Config::instance().sts_config.beta);

		cout << "done" << endl;
	}
	else {
		cout << "Loading networks...";

		load(p_timestamp);

		Config::instance().f5_config.dim_x = _STS->dim_x();
		Config::instance().f5_config.dim_y = _STS->dim_y();
		Config::instance().sts_config.dim_x = _F5->dim_x();
		Config::instance().sts_config.dim_y = _F5->dim_y();

		cout << "done" << endl;
	}
}

void ModelMNS::run(const int p_epochs) {
	cout << "Epochs: " << p_epochs << endl;
	cout << "Settling: " << Config::instance().settling << endl;
	cout << "CPUs: " << GetProcessorCount() << endl;
	cout << "Initializing learning module...";

	SOM_analyzer F5_analyzer;
	SOM_analyzer STS_analyzer;

	MSOM_params F5_params(_F5);
	F5_params.init_training(Config::instance().f5_config.gamma1, Config::instance().f5_config.gamma2, p_epochs);

	MSOM_params STS_params(_STS);
	STS_params.init_training(Config::instance().sts_config.gamma1, Config::instance().sts_config.gamma2, p_epochs);

	MSOM_learning F5_learner(_F5, &F5_params, &F5_analyzer);
	MSOM_learning STS_learner(_STS, &STS_params, &STS_analyzer);

	vector<Sequence*>* trainData = _data.permute();

	vector<MSOM_learning*> F5_thread(trainData->size());
	vector<SOM_analyzer*> F5_thread_analyzer(trainData->size());

	for (int i = 0; i < trainData->size(); i++) {
		F5_thread_analyzer[i] = new SOM_analyzer();
		F5_thread[i] = new MSOM_learning(_F5->clone(), &F5_params, F5_thread_analyzer[i]);
	}

	vector<MSOM_learning*> STS_thread(trainData->size());
	vector<SOM_analyzer*> STS_thread_analyzer(trainData->size());

	for (int i = 0; i < trainData->size(); i++) {
		STS_thread_analyzer[i] = new SOM_analyzer();
		STS_thread[i] = new MSOM_learning(_STS->clone(), &STS_params, STS_thread_analyzer[i]);
	}

	cout << "done" << endl;

	for (int t = 0; t < p_epochs; t++) {
		cout << "Epoch " << t << endl;
		trainData = _data.permute();

		const auto start = chrono::system_clock::now();

		parallel_for(0, static_cast<int>(trainData->size()), [&](int i) {
			Tensor f5_input = Tensor::Zero({ _sizeF5input + Config::instance().sts_config.dim_x * Config::instance().sts_config.dim_y });
			Tensor sts_input = Tensor::Zero({ _sizeSTSinput + Config::instance().f5_config.dim_x * Config::instance().f5_config.dim_y });

			F5_thread[i]->init_msom(_F5);

			for (int p = 0; p < PERSPS; p++) {
				STS_thread[i]->init_msom(_STS);

				for (int j = 0; j < trainData->at(i)->getMotorData()->size(); j++) {
					Tensor* motor_sample = trainData->at(i)->getMotorData()->at(j);
					Tensor* visual_sample = trainData->at(i)->getVisualData(p)->at(j);

					F5_thread[i]->train(motor_sample);
					STS_thread[i]->train(visual_sample);
				}

				F5_thread[i]->msom()->reset_context();
				STS_thread[i]->msom()->reset_context();
			}
		});

		F5_learner.merge(F5_thread);
		STS_learner.merge(STS_thread);
		F5_analyzer.merge(F5_thread_analyzer);
		STS_analyzer.merge(STS_thread_analyzer);

		const auto end = chrono::system_clock::now();
		chrono::duration<double> elapsed_seconds = end - start;
		cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
		cout << " PMC qError: " << F5_analyzer.q_error() << " WD: " << F5_analyzer.winner_diff(_F5->get_lattice()->getDim()) << endl;
		cout << "STSp qError: " << STS_analyzer.q_error() << " WD: " << STS_analyzer.winner_diff(_STS->get_lattice()->getDim()) << endl;
		F5_analyzer.end_epoch();
		STS_analyzer.end_epoch();
		F5_params.param_decay();
		STS_params.param_decay();
	}

	for (int i = 0; i < trainData->size(); i++) {
		delete F5_thread[i];
		delete F5_thread_analyzer[i];
	}
	for (int i = 0; i < trainData->size(); i++) {
		delete STS_thread[i];
		delete STS_thread_analyzer[i];
	}
}

void ModelMNS::save() const {
	const string timestamp = to_string(time(nullptr));

	IOUtils::save_network(timestamp + "_F5.json", _F5);
	IOUtils::save_network(timestamp + "_STS.json", _STS);
}

void ModelMNS::load(const string p_timestamp) {
	_F5 = static_cast<MSOM*>(IOUtils::load_network("C:\\GIT\\Coeus\\x64\\Debug\\" + p_timestamp + "_F5.json"));
	_STS = static_cast<MSOM*>(IOUtils::load_network("C:\\GIT\\Coeus\\x64\\Debug\\" + p_timestamp + "_STS.json"));
}

void ModelMNS::save_results(const string p_filename, const int p_dim_x, const int p_dim_y, double* p_data, const int p_category) {
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

void ModelMNS::testDistance() {
	const string timestamp = to_string(time(nullptr));

	vector<Sequence*>* trainData = _data.permute();

	double* winRateF5_Motor = static_cast<double*>(calloc(Config::instance().f5_config.dim_x * Config::instance().f5_config.dim_y * GRASPS, sizeof(double)));
	double* winRateF5_Visual = static_cast<double*>(calloc(Config::instance().f5_config.dim_x * Config::instance().f5_config.dim_y * PERSPS, sizeof(double)));
	double* winRateSTS_Visual = static_cast<double*>(calloc(Config::instance().sts_config.dim_x * Config::instance().sts_config.dim_y * PERSPS, sizeof(double)));
	double* winRateSTS_Motor = static_cast<double*>(calloc(Config::instance().sts_config.dim_x * Config::instance().sts_config.dim_y * GRASPS, sizeof(double)));

	Tensor f5_input = Tensor::Zero({ _sizeF5input + Config::instance().sts_config.dim_x * Config::instance().sts_config.dim_y });
	Tensor sts_input = Tensor::Zero({ _sizeSTSinput + Config::instance().f5_config.dim_x * Config::instance().f5_config.dim_y });

	for (int i = 0; i < trainData->size(); i++) {
		for (int p = 0; p < PERSPS; p++) {
			for (int j = 0; j < trainData->at(i)->getMotorData()->size(); j++) {
				Tensor* motor_sample = trainData->at(i)->getMotorData()->at(j);
				Tensor* visual_sample = trainData->at(i)->getVisualData(p)->at(j);

				_F5->activate(motor_sample);
				_STS->activate(visual_sample);
			}

			for (int n = 0; n < _STS->get_lattice()->getDim(); n++) {
				winRateSTS_Visual[n * PERSPS + p] += _STS->get_output()->at(n);
				winRateSTS_Motor[n * GRASPS + trainData->at(i)->getGrasp() - 1] += _STS->get_output()->at(n);
			}


			for (int n = 0; n < _F5->get_lattice()->getDim(); n++) {
				winRateF5_Visual[n * PERSPS + p] += _F5->get_output()->at(n);
				winRateF5_Motor[n * GRASPS + trainData->at(i)->getGrasp() - 1] += _F5->get_output()->at(n);
			}
		}
	}

	save_results(timestamp + "_F5.mot", Config::instance().f5_config.dim_x, Config::instance().f5_config.dim_y, winRateF5_Motor, GRASPS);
	save_results(timestamp + "_F5.vis", Config::instance().f5_config.dim_x, Config::instance().f5_config.dim_y, winRateF5_Visual, PERSPS);
	save_results(timestamp + "_STS.mot", Config::instance().sts_config.dim_x, Config::instance().sts_config.dim_y, winRateSTS_Motor, GRASPS);
	save_results(timestamp + "_STS.vis", Config::instance().sts_config.dim_x, Config::instance().sts_config.dim_y, winRateSTS_Visual, PERSPS);
}

void ModelMNS::testFinalWinners() {
	const string timestamp = to_string(time(nullptr));

	vector<Sequence*>* trainData = _data.permute();

	double* winRateF5_Motor = static_cast<double*>(calloc(Config::instance().f5_config.dim_x * Config::instance().f5_config.dim_y * GRASPS, sizeof(double)));
	double* winRateF5_Visual = static_cast<double*>(calloc(Config::instance().f5_config.dim_x * Config::instance().f5_config.dim_y * PERSPS, sizeof(double)));
	double* winRateSTS_Visual = static_cast<double*>(calloc(Config::instance().sts_config.dim_x * Config::instance().sts_config.dim_y * PERSPS, sizeof(double)));
	double* winRateSTS_Motor = static_cast<double*>(calloc(Config::instance().sts_config.dim_x * Config::instance().sts_config.dim_y * GRASPS, sizeof(double)));

	Tensor f5_input = Tensor::Zero({ _sizeF5input + Config::instance().sts_config.dim_x * Config::instance().sts_config.dim_y });
	Tensor sts_input = Tensor::Zero({ _sizeSTSinput + Config::instance().f5_config.dim_x * Config::instance().f5_config.dim_y });

	for (int i = 0; i < trainData->size(); i++) {
		for (int p = 0; p < PERSPS; p++) {
			for (int j = 0; j < trainData->at(i)->getMotorData()->size(); j++) {
				Tensor* motor_sample = trainData->at(i)->getMotorData()->at(j);
				Tensor* visual_sample = trainData->at(i)->getVisualData(p)->at(j);

				_F5->activate(motor_sample);
				_STS->activate(visual_sample);
			}

			_F5->reset_context();
			_STS->reset_context();

			winRateSTS_Visual[_STS->get_winner() * PERSPS + p]++;
			winRateSTS_Motor[_STS->get_winner() * GRASPS + trainData->at(i)->getGrasp() - 1]++;

			winRateF5_Visual[_F5->get_winner() * PERSPS + p]++;
			winRateF5_Motor[_F5->get_winner() * GRASPS + trainData->at(i)->getGrasp() - 1]++;
		}
	}

	save_results(timestamp + "_F5.mot", Config::instance().f5_config.dim_x, Config::instance().f5_config.dim_y, winRateF5_Motor, GRASPS);
	save_results(timestamp + "_F5.vis", Config::instance().f5_config.dim_x, Config::instance().f5_config.dim_y, winRateF5_Visual, PERSPS);
	save_results(timestamp + "_STS.mot", Config::instance().sts_config.dim_x, Config::instance().sts_config.dim_y, winRateSTS_Motor, GRASPS);
	save_results(timestamp + "_STS.vis", Config::instance().sts_config.dim_x, Config::instance().sts_config.dim_y, winRateSTS_Visual, PERSPS);
}