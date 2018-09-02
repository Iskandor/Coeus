//
// Created by user on 14. 12. 2017.
//

#include <fstream>
#include <chrono>
#include "ModelMNS3.h"
#include "MSOM_learning.h"
#include "SOM_learning.h"
#include <iostream>
#include <string>
#include "IOUtils.h"
#include <ppl.h>
#include <concrtrm.h>
#include "Config.h"
#include "Logger.h"

using namespace MNS;
using namespace Concurrency;
using namespace std;

ModelMNS3::ModelMNS3() {
	_F5 = nullptr;
	_STS = nullptr;

	_f5_mask_pre = nullptr;
	_f5_mask_post = nullptr;
	_sts_mask = nullptr;
}

ModelMNS3::~ModelMNS3() {
    delete _F5;
    delete _STS;

	int len = _data.permute()->size();

	delete[] _f5_mask_pre;
	delete[] _f5_mask_post;
	delete[] _sts_mask;

}

void ModelMNS3::init(string p_timestamp) {
	cout << "Loading data...";
    _data.loadData( Config::instance().visual_data, Config::instance().motor_data);
	cout << "done" << endl;

	if (p_timestamp.empty()) {
		cout << "Initializing networks...";

		_F5 = new MSOM(
			"F5",
			_sizeF5input + Config::instance().sts_config.dim_x * Config::instance().sts_config.dim_y, 
			Config::instance().f5_config.dim_x,
			Config::instance().f5_config.dim_y,
			EXPONENTIAL, 
			Config::instance().f5_config.alpha, 
			Config::instance().f5_config.beta);

		_STS = new MSOM(
			"STS",
			_sizeSTSinput + Config::instance().f5_config.dim_x * Config::instance().f5_config.dim_y,
			Config::instance().sts_config.dim_x, 
			Config::instance().sts_config.dim_y, 
			EXPONENTIAL, 
			Config::instance().sts_config.alpha, 
			Config::instance().sts_config.beta);

		cout << "done" << endl;
	}
	else {
		cout << "Loading networks...";

		load(p_timestamp);

		Config::instance().f5_config.dim_x = _F5->dim_x();
		Config::instance().f5_config.dim_y = _F5->dim_y();
		Config::instance().sts_config.dim_x = _STS->dim_x();
		Config::instance().sts_config.dim_y = _STS->dim_y();

		cout << "done" << endl;
	}

	cout << "Initializing masks...";

	int len = _data.permute()->size();

	_f5_mask_post = new int[_sizeF5input + Config::instance().sts_config.dim_x * Config::instance().sts_config.dim_y];
	_f5_mask_pre = new int[_sizeF5input + Config::instance().sts_config.dim_x * Config::instance().sts_config.dim_y];
	_sts_mask = new int[_sizeSTSinput + Config::instance().f5_config.dim_x * Config::instance().f5_config.dim_y];

	for (int i = 0; i < _sizeF5input + Config::instance().sts_config.dim_x * Config::instance().sts_config.dim_y; i++) {
		if (i < _sizeF5input) {
			_f5_mask_pre[i] = 1;
			_f5_mask_post[i] = 0;
		}
		else {
			_f5_mask_pre[i] = 0;
			_f5_mask_post[i] = 1;
		}
	}

	for (int i = 0; i < _sizeSTSinput + Config::instance().f5_config.dim_x * Config::instance().f5_config.dim_y; i++) {
		if (i < _sizeSTSinput) {
			_sts_mask[i] = 1;
		}
		else {
			_sts_mask[i] = 0;
		}
	}

	cout << "done" << endl;
}

void ModelMNS3::run(const int p_epochs) {
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
					

					F5_thread[i]->msom()->set_input_mask(_f5_mask_pre);
					prepareInputF5(&f5_input, motor_sample, STS_thread[i]->msom());
					F5_thread[i]->msom()->activate(&f5_input);
					F5_thread[i]->msom()->set_input_mask(nullptr);

					STS_thread[i]->msom()->set_input_mask(_sts_mask);
					prepareInputSTS(&sts_input, visual_sample, F5_thread[i]->msom());
					STS_thread[i]->msom()->activate(&sts_input);
					STS_thread[i]->msom()->set_input_mask(nullptr);

					for (int s = 0; s < Config::instance().settling; s++) {
						prepareInputF5(&f5_input, motor_sample, STS_thread[i]->msom());
						prepareInputSTS(&sts_input, visual_sample, F5_thread[i]->msom());
						F5_thread[i]->train(&f5_input);
						STS_thread[i]->train(&sts_input);
						F5_thread[i]->msom()->activate(&f5_input);
						STS_thread[i]->msom()->activate(&sts_input);
					}
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

		double qerrF5 = F5_analyzer.q_error(_F5->get_input_group()->get_dim());
		double qerrSTS = STS_analyzer.q_error(_STS->get_input_group()->get_dim());
		double wdF5 = F5_analyzer.winner_diff(_F5->get_lattice()->get_dim());
		double wdSTS = STS_analyzer.winner_diff(_STS->get_lattice()->get_dim());

		cout << " F5 qError: " << qerrF5 << " WD: " << wdF5 << endl;
		cout << "STS qError: " << qerrSTS << " WD: " << wdSTS << endl;

		Logger::instance().log(to_string(qerrF5) + "," + to_string(wdF5) + "," + to_string(qerrSTS) + "," + to_string(wdSTS));

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

void ModelMNS3::run2(const int p_epochs) {
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


					F5_thread[i]->msom()->set_input_mask(_f5_mask_pre);
					prepareInputF5(&f5_input, motor_sample, STS_thread[i]->msom());
					F5_thread[i]->msom()->activate(&f5_input);
					F5_thread[i]->msom()->set_input_mask(nullptr);

					STS_thread[i]->msom()->set_input_mask(_sts_mask);
					prepareInputSTS(&sts_input, visual_sample, F5_thread[i]->msom());
					STS_thread[i]->msom()->activate(&sts_input);
					STS_thread[i]->msom()->set_input_mask(nullptr);

					for (int s = 0; s < Config::instance().settling; s++) {
						prepareInputF5(&f5_input, motor_sample, STS_thread[i]->msom());
						prepareInputSTS(&sts_input, visual_sample, F5_thread[i]->msom());
						F5_thread[i]->msom()->activate(&f5_input);
						STS_thread[i]->msom()->activate(&sts_input);
					}

					prepareInputF5(&f5_input, motor_sample, STS_thread[i]->msom());
					prepareInputSTS(&sts_input, visual_sample, F5_thread[i]->msom());
					F5_thread[i]->train(&f5_input);
					STS_thread[i]->train(&sts_input);
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

		double qerrF5 = F5_analyzer.q_error(_F5->get_input_group()->get_dim());
		double qerrSTS = STS_analyzer.q_error(_STS->get_input_group()->get_dim());
		double wdF5 = F5_analyzer.winner_diff(_F5->get_lattice()->get_dim());
		double wdSTS = STS_analyzer.winner_diff(_STS->get_lattice()->get_dim());

		cout << " F5 qError: " << qerrF5 << " WD: " << wdF5 << endl;
		cout << "STS qError: " << qerrSTS << " WD: " << wdSTS << endl;

		Logger::instance().log(to_string(qerrF5) + "," + to_string(wdF5) + "," + to_string(qerrSTS) + "," + to_string(wdSTS));

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

void ModelMNS3::save_umatrix(string p_timestamp)
{
	SOM_analyzer analyzer;

	analyzer.create_umatrix(_F5);
	analyzer.save_umatrix(p_timestamp + "_F5.umat");

	analyzer.create_umatrix(_STS);
	analyzer.save_umatrix(p_timestamp + "_STS.umat");
}

void ModelMNS3::save(string p_timestamp) const {
    IOUtils::save_network(p_timestamp + "_F5.json", _F5);
	IOUtils::save_network(p_timestamp + "_STS.json", _STS);
}

void ModelMNS3::load(const string p_timestamp) {
    _F5 = static_cast<MSOM*>(IOUtils::load_network(p_timestamp + "_F5.json"));
    _STS = static_cast<MSOM*>(IOUtils::load_network(p_timestamp + "_STS.json"));
}

void ModelMNS3::prepareInputSTS(Tensor* p_output, Tensor *p_input, MSOM* p_f5) const {
    Tensor::Concat(p_output, p_input, p_f5->get_output());
}

void ModelMNS3::prepareInputF5(Tensor* p_output, Tensor *p_input, MSOM* p_sts) const {
    Tensor::Concat(p_output, p_input, p_sts->get_output());
}

void ModelMNS3::save_results(const string p_filename, const int p_dim_x, const int p_dim_y, double* p_data, const int p_category) {
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

void ModelMNS3::testDistance() {
    const string timestamp = to_string(time(nullptr));

    vector<Sequence*>* trainData = _data.permute();

    double* winRateF5_Motor = static_cast<double*>(calloc(Config::instance().f5_config.dim_x * Config::instance().f5_config.dim_y * GRASPS, sizeof(double)));
	double* winRateF5_Visual = static_cast<double*>(calloc(Config::instance().f5_config.dim_x * Config::instance().f5_config.dim_y * PERSPS, sizeof(double)));
    double* winRateSTS_Visual = static_cast<double*>(calloc(Config::instance().sts_config.dim_x * Config::instance().sts_config.dim_y * PERSPS, sizeof(double)));
    double* winRateSTS_Motor = static_cast<double*>(calloc(Config::instance().sts_config.dim_x * Config::instance().sts_config.dim_y * GRASPS, sizeof(double)));

	SOM_analyzer analyzerF5;
	SOM_analyzer analyzerSTS;

	Tensor f5_input = Tensor::Zero({ _sizeF5input + Config::instance().sts_config.dim_x * Config::instance().sts_config.dim_y });
	Tensor sts_input = Tensor::Zero({ _sizeSTSinput + Config::instance().f5_config.dim_x * Config::instance().f5_config.dim_y });

	for (int i = 0; i < trainData->size(); i++) {
		for (int p = 0; p < PERSPS; p++) {
			for (int j = 0; j < trainData->at(i)->getMotorData()->size(); j++) {
				Tensor* motor_sample = trainData->at(i)->getMotorData()->at(j);
				Tensor* visual_sample = trainData->at(i)->getVisualData(p)->at(j);

				_F5->set_input_mask(_f5_mask_pre);
				prepareInputF5(&f5_input, motor_sample, _STS);
				_F5->activate(&f5_input);
				_F5->set_input_mask(nullptr);

				_STS->set_input_mask(_sts_mask);
				prepareInputSTS(&sts_input, visual_sample, _F5);
				_STS->activate(&sts_input);
				_STS->set_input_mask(nullptr);

				for (int s = 0; s < Config::instance().settling; s++) {
					prepareInputF5(&f5_input, motor_sample, _STS);
					prepareInputSTS(&sts_input, visual_sample, _F5);
					_F5->activate(&f5_input);
					_STS->activate(&sts_input);
				}

				analyzerF5.update(_F5, _F5->get_winner());
				analyzerSTS.update(_STS, _STS->get_winner());
			}

			_F5->reset_context();
			_STS->reset_context();

			for (int n = 0; n < _STS->get_lattice()->get_dim(); n++) {
				winRateSTS_Visual[n * PERSPS + p] += _STS->get_output()->at(n);
				winRateSTS_Motor[n * GRASPS + trainData->at(i)->getGrasp() - 1] += _STS->get_output()->at(n);
			}


			for (int n = 0; n < _F5->get_lattice()->get_dim(); n++) {
				winRateF5_Visual[n * PERSPS + p] += _F5->get_output()->at(n);
				winRateF5_Motor[n * GRASPS + trainData->at(i)->getGrasp() - 1] += _F5->get_output()->at(n);
			}
		}
	}

	cout << " F5 qError: " << analyzerF5.q_error(_F5->get_input_group()->get_dim()) << " WD: " << analyzerF5.winner_diff(_F5->get_lattice()->get_dim()) << endl;
	cout << " STS qError: " << analyzerSTS.q_error(_STS->get_input_group()->get_dim()) << " WD: " << analyzerSTS.winner_diff(_STS->get_lattice()->get_dim()) << endl;

	save_results(timestamp + "_F5.mot", Config::instance().f5_config.dim_x, Config::instance().f5_config.dim_y, winRateF5_Motor, GRASPS);
	save_results(timestamp + "_F5.vis", Config::instance().f5_config.dim_x, Config::instance().f5_config.dim_y, winRateF5_Visual, PERSPS);
	save_results(timestamp + "_STS.mot", Config::instance().sts_config.dim_x, Config::instance().sts_config.dim_y, winRateSTS_Motor, GRASPS);
	save_results(timestamp + "_STS.vis", Config::instance().sts_config.dim_x, Config::instance().sts_config.dim_y, winRateSTS_Visual, PERSPS);
}

void ModelMNS3::testFinalWinners() {
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

				_F5->set_input_mask(_f5_mask_pre);
				prepareInputF5(&f5_input, motor_sample, _STS);
				_F5->activate(&f5_input);
				_F5->set_input_mask(nullptr);

				_STS->set_input_mask(_sts_mask);
				prepareInputSTS(&sts_input, visual_sample, _F5);
				_STS->activate(&sts_input);
				_STS->set_input_mask(nullptr);

				for (int s = 0; s < Config::instance().settling; s++) {
					prepareInputF5(&f5_input, motor_sample, _STS);
					prepareInputSTS(&sts_input, visual_sample, _F5);
					_F5->activate(&f5_input);
					_STS->activate(&sts_input);
				}
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

void ModelMNS3::testMirror(int p_persp) {
	const string timestamp = to_string(time(nullptr));

	vector<Sequence*>* trainData = _data.permute();

	double* winRateF5_Motor = static_cast<double*>(calloc(Config::instance().f5_config.dim_x * Config::instance().f5_config.dim_y * GRASPS, sizeof(double)));
	double* winRateF5_Visual = static_cast<double*>(calloc(Config::instance().f5_config.dim_x * Config::instance().f5_config.dim_y * PERSPS, sizeof(double)));
	double* winRateSTS_Visual = static_cast<double*>(calloc(Config::instance().sts_config.dim_x * Config::instance().sts_config.dim_y * PERSPS, sizeof(double)));
	double* winRateSTS_Motor = static_cast<double*>(calloc(Config::instance().sts_config.dim_x * Config::instance().sts_config.dim_y * GRASPS, sizeof(double)));

	Tensor f5_input = Tensor::Zero({ _sizeF5input + Config::instance().sts_config.dim_x * Config::instance().sts_config.dim_y });
	Tensor sts_input = Tensor::Zero({ _sizeSTSinput + Config::instance().f5_config.dim_x * Config::instance().f5_config.dim_y });

	for (int i = 0; i < 1; i++) {
		for (int j = 0; j < trainData->at(i)->getMotorData()->size(); j++) {
			Tensor* motor_sample = trainData->at(i)->getMotorData()->at(j);
			Tensor* visual_sample = trainData->at(i)->getVisualData(p_persp)->at(j);

			_STS->set_input_mask(_sts_mask);
			prepareInputSTS(&sts_input, visual_sample, _F5);
			_STS->activate(&sts_input);
			_STS->set_input_mask(nullptr);

			_F5->set_input_mask(_f5_mask_post);
			prepareInputF5(&f5_input, motor_sample, _STS);
			_F5->activate(&f5_input);

			for (int s = 0; s < Config::instance().settling; s++) {
				prepareInputF5(&f5_input, motor_sample, _STS);
				prepareInputSTS(&sts_input, visual_sample, _F5);
				_F5->activate(&f5_input);
				_STS->activate(&sts_input);
			}
		}

		_F5->reset_context();
		_STS->reset_context();

		for (int n = 0; n < _STS->get_lattice()->get_dim(); n++) {
			winRateSTS_Visual[n * PERSPS + p_persp] += _STS->get_output()->at(n);
			winRateSTS_Motor[n * GRASPS + trainData->at(i)->getGrasp() - 1] += _STS->get_output()->at(n);
		}
		
			
		for (int n = 0; n < _F5->get_lattice()->get_dim(); n++) {
			winRateF5_Visual[n * PERSPS + p_persp] += _F5->get_output()->at(n);
			winRateF5_Motor[n * GRASPS + trainData->at(i)->getGrasp() - 1] += _F5->get_output()->at(n);
		}
	}

	save_results(timestamp + "_F5.mot", Config::instance().f5_config.dim_x, Config::instance().f5_config.dim_y, winRateF5_Motor, GRASPS);
	save_results(timestamp + "_F5.vis", Config::instance().f5_config.dim_x, Config::instance().f5_config.dim_y, winRateF5_Visual, PERSPS);
	save_results(timestamp + "_STS.mot", Config::instance().sts_config.dim_x, Config::instance().sts_config.dim_y, winRateSTS_Motor, GRASPS);
	save_results(timestamp + "_STS.vis", Config::instance().sts_config.dim_x, Config::instance().sts_config.dim_y, winRateSTS_Visual, PERSPS);
}