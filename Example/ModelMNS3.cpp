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
#include "Config.h"

using namespace MNS;
using namespace Concurrency;
using namespace std;

ModelMNS3::ModelMNS3() {
    _F5 = nullptr;
    _STS = nullptr;
}

ModelMNS3::~ModelMNS3() {
    delete _F5;
    delete _STS;
}

void ModelMNS3::init() {
    _data.loadData("../data/Trajectories.3.vd", "../data/Trajectories.3.md");

    _F5 = new MSOM(_sizeF5input + _sizeSTS * _sizeSTS, _sizeF5, _sizeF5, NeuralGroup::EXPONENTIAL, Config::instance().f5_config.alpha, Config::instance().f5_config.beta);
    _STS = new MSOM(_sizeSTSinput + _sizeF5 * _sizeF5, _sizeSTS, _sizeSTS, NeuralGroup::EXPONENTIAL, Config::instance().sts_config.alpha, Config::instance().sts_config.beta);

	for(int i = 0; i < _sizeF5input + _sizeSTS * _sizeSTS; i++) {
		if (i < _sizeF5input) {
			_f5_mask_pre[i] = 1;
			_f5_mask_post[i] = 0;
		}
		else {
			_f5_mask_pre[i] = 0;
			_f5_mask_post[i] = 1;
		}
	}

	for (int i = 0; i < _sizeSTSinput + _sizeF5 * _sizeF5; i++) {
		if (i < _sizeSTSinput) {
			_sts_mask[i] = 1;
		}
		else {
			_sts_mask[i] = 0;
		}
	}

	vector<Sequence*>* trainData = _data.permute();

	_F5input = new Tensor*[trainData->size()];
	_STSinput = new Tensor*[trainData->size() * PERSPS];

	for (int i = 0; i < trainData->size(); i++) {
		_F5input[i] = new Tensor({ _sizeF5input + _sizeSTS * _sizeSTS }, Tensor::ZERO);
	}
	for (int i = 0; i < trainData->size() * PERSPS; i++) {
		_STSinput[i] = new Tensor({ _sizeSTSinput + _sizeF5 * _sizeF5 }, Tensor::ZERO);
	}
}

void ModelMNS3::run(int p_epochs) {
	cout << "Epochs: " << p_epochs << endl;
	cout << "Settling: " << Config::instance().settling << endl;

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

	for (int i = 0; i < trainData->size(); i++) {
		F5_thread[i] = new MSOM_learning(_F5->clone(), &F5_params, &F5_analyzer);
	}

	vector<MSOM_learning*> STS_thread(trainData->size() * PERSPS);

	for (int i = 0; i < trainData->size() * PERSPS; i++) {
		STS_thread[i] = new MSOM_learning(_STS->clone(), &STS_params, &STS_analyzer);
	}

	for (int t = 0; t < p_epochs; t++) {
		cout << "Epoch " << t << endl;
		trainData = _data.permute();

		const auto start = chrono::system_clock::now();

		parallel_for(0, static_cast<int>(trainData->size()), [&](int i) {
			for (int p = 0; p < PERSPS; p++) {
				F5_thread[i]->init_msom(_F5);
				STS_thread[i * PERSPS + p]->init_msom(_STS);

				F5_thread[i]->msom()->set_input_mask(_f5_mask_pre);
				activateF5(i, F5_thread[i]->msom(), trainData->at(i)->getMotorData());
				F5_thread[i]->msom()->set_input_mask(nullptr);

				STS_thread[i * PERSPS + p]->msom()->set_input_mask(_sts_mask);
				activateSTS(i, STS_thread[i * PERSPS + p]->msom(), trainData->at(i)->getVisualData(p));
				STS_thread[i * PERSPS + p]->msom()->set_input_mask(nullptr);

				for(int s = 0; s < Config::instance().settling; s++) {
					trainF5(i, F5_thread[i], trainData->at(i)->getMotorData());
					trainSTS(i * PERSPS + p, STS_thread[i * PERSPS + p], trainData->at(i)->getVisualData(p));
				}
			}
		});

		F5_learner.merge(F5_thread);
		STS_learner.merge(STS_thread);

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
}

void ModelMNS3::save() {
    const string timestamp = to_string(time(nullptr));

    IOUtils::save_network(timestamp + "_F5.json", _F5);
	IOUtils::save_network(timestamp + "_STS.json", _STS);
}

void ModelMNS3::load(const string p_timestamp) {
    _F5 = static_cast<MSOM*>(IOUtils::load_network("C:\\GIT\\Coeus\\x64\\Debug\\" + p_timestamp + "_F5.json"));
    _STS = static_cast<MSOM*>(IOUtils::load_network("C:\\GIT\\Coeus\\x64\\Debug\\" + p_timestamp + "_STS.json"));
}

void ModelMNS3::prepareInputSTS(int p_index, Tensor *p_input) {
    Tensor::Concat(_STSinput[p_index], p_input, _F5->get_output());
}

void ModelMNS3::prepareInputF5(int p_index, Tensor *p_input) {
    Tensor::Concat(_F5input[p_index], p_input, _STS->get_output());
}

void ModelMNS3::save_results(const string p_filename, const int p_dim_x, const int p_dim_y, double* p_data, const int p_category) const {
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

    double* winRateF5_Motor = static_cast<double*>(calloc(_sizeF5 * _sizeF5 * GRASPS, sizeof(double)));
	double* winRateF5_Visual = static_cast<double*>(calloc(_sizeF5 * _sizeF5 * PERSPS, sizeof(double)));
    double* winRateSTS_Visual = static_cast<double*>(calloc(_sizeSTS * _sizeSTS * PERSPS, sizeof(double)));
    double* winRateSTS_Motor = static_cast<double*>(calloc(_sizeSTS * _sizeSTS * GRASPS, sizeof(double)));

	for (int i = 0; i < trainData->size(); i++) {
		_F5->set_input_mask(_f5_mask_pre);
		activateF5(0, _F5, trainData->at(i)->getMotorData());
		_F5->set_input_mask(nullptr);

		for (int p = 0; p < PERSPS; p++) {
			_STS->set_input_mask(_sts_mask);
			activateSTS(0, _STS, trainData->at(i)->getVisualData(p));
			_STS->set_input_mask(nullptr);

			for(int s = 0; s < 5; s++) {
				activateSTS(0, _STS, trainData->at(i)->getVisualData(p));

				activateF5(0, _F5, trainData->at(i)->getMotorData());
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

	save_results(timestamp + "_F5.mot", _sizeF5, _sizeF5, winRateF5_Motor, GRASPS);
	save_results(timestamp + "_F5.vis", _sizeF5, _sizeF5, winRateF5_Visual, PERSPS);
	save_results(timestamp + "_STS.mot", _sizeSTS, _sizeSTS, winRateSTS_Motor, GRASPS);
	save_results(timestamp + "_STS.vis", _sizeSTS, _sizeSTS, winRateSTS_Visual, PERSPS);
}

void ModelMNS3::testFinalWinners() {
    const string timestamp = to_string(time(nullptr));

    vector<Sequence*>* trainData = _data.permute();

	double* winRateF5_Motor = static_cast<double*>(calloc(_sizeF5 * _sizeF5 * GRASPS, sizeof(double)));
	double* winRateF5_Visual = static_cast<double*>(calloc(_sizeF5 * _sizeF5 * PERSPS, sizeof(double)));
	double* winRateSTS_Visual = static_cast<double*>(calloc(_sizeSTS * _sizeSTS * PERSPS, sizeof(double)));
	double* winRateSTS_Motor = static_cast<double*>(calloc(_sizeSTS * _sizeSTS * GRASPS, sizeof(double)));
 
	for (int i = 0; i < trainData->size(); i++) {
		_F5->set_input_mask(_f5_mask_pre);
		activateF5(0, _F5, trainData->at(i)->getMotorData());
		_F5->set_input_mask(nullptr);

		for (int p = 0; p < PERSPS; p++) {
			_STS->set_input_mask(_sts_mask);
			activateSTS(0, _STS, trainData->at(i)->getVisualData(p));
			_STS->set_input_mask(nullptr);

			for (int j = 0; j < trainData->at(i)->getVisualData(p)->size(); j++) {
				prepareInputSTS(0, trainData->at(i)->getVisualData(p)->at(j));
				_STS->activate(_STSinput[0]);
			}
			winRateSTS_Visual[_STS->get_winner() * PERSPS + p]++;
			winRateSTS_Motor[_STS->get_winner() * GRASPS + trainData->at(i)->getGrasp() - 1]++;
			_STS->reset_context();

			for (int j = 0; j < trainData->at(i)->getMotorData()->size(); j++) {
				prepareInputF5(0, trainData->at(i)->getMotorData()->at(j));
				_F5->activate(_F5input[0]);
			}
			winRateF5_Visual[_F5->get_winner() * PERSPS + p]++;
			winRateF5_Motor[_F5->get_winner() * GRASPS + trainData->at(i)->getGrasp() - 1]++;
			_F5->reset_context();
		}
	}

	save_results(timestamp + "_F5.mot", _sizeF5, _sizeF5, winRateF5_Motor, GRASPS);
	save_results(timestamp + "_F5.vis", _sizeF5, _sizeF5, winRateF5_Visual, PERSPS);
	save_results(timestamp + "_STS.mot", _sizeSTS, _sizeSTS, winRateSTS_Motor, GRASPS);
	save_results(timestamp + "_STS.vis", _sizeSTS, _sizeSTS, winRateSTS_Visual, PERSPS);
}

void ModelMNS3::testMirror() {
	const string timestamp = to_string(time(nullptr));

	vector<Sequence*>* trainData = _data.permute();

	double* winRateF5_Motor = static_cast<double*>(calloc(_sizeF5 * _sizeF5 * GRASPS, sizeof(double)));
	double* winRateF5_Visual = static_cast<double*>(calloc(_sizeF5 * _sizeF5 * PERSPS, sizeof(double)));
	double* winRateSTS_Visual = static_cast<double*>(calloc(_sizeSTS * _sizeSTS * PERSPS, sizeof(double)));
	double* winRateSTS_Motor = static_cast<double*>(calloc(_sizeSTS * _sizeSTS * GRASPS, sizeof(double)));

	for (int i = 0; i < 1; i++) {
		for (int p = 0; p < 1; p++) {
			_STS->set_input_mask(_sts_mask);
			activateSTS(0, _STS, trainData->at(i)->getVisualData(p));
			_F5->set_input_mask(_f5_mask_post);
			activateF5(0, _F5, trainData->at(i)->getMotorData());
			_STS->set_input_mask(nullptr);
			activateSTS(0, _STS, trainData->at(i)->getVisualData(p));

			for(int s = 0; s < 5; s++) {
				_F5->set_input_mask(_f5_mask_post);
				activateF5(0, _F5, trainData->at(i)->getMotorData());
				_STS->set_input_mask(nullptr);
				activateSTS(0, _STS, trainData->at(i)->getVisualData(p));
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

	save_results(timestamp + "_F5.mot", _sizeF5, _sizeF5, winRateF5_Motor, GRASPS);
	save_results(timestamp + "_F5.vis", _sizeF5, _sizeF5, winRateF5_Visual, PERSPS);
	save_results(timestamp + "_STS.mot", _sizeSTS, _sizeSTS, winRateSTS_Motor, GRASPS);
	save_results(timestamp + "_STS.vis", _sizeSTS, _sizeSTS, winRateSTS_Visual, PERSPS);
}

void MNS::ModelMNS3::activateF5(int p_index, MSOM* p_msom, vector<Tensor*>* p_input)
{
	for (int j = 0; j < p_input->size(); j++) {
		prepareInputF5(p_index, p_input->at(j));
		p_msom->activate(_F5input[p_index]);
	}
	p_msom->reset_context();

}

void MNS::ModelMNS3::activateSTS(int p_index, MSOM* p_msom, vector<Tensor*>* p_input)
{
	for (int j = 0; j < p_input->size(); j++) {
		prepareInputSTS(p_index, p_input->at(j));
		p_msom->activate(_STSinput[p_index]);
	}
	p_msom->reset_context();
}

void ModelMNS3::trainF5(int p_index, MSOM_learning* p_F5_learner, vector<Tensor*>* p_input) {
	for (int j = 0; j < p_input->size(); j++) {
		prepareInputF5(p_index, p_input->at(j));
		p_F5_learner->train(_F5input[p_index]);
	}
	p_F5_learner->reset_context();
}

void ModelMNS3::trainSTS(int p_index, MSOM_learning* p_STS_learner, vector<Tensor*>* p_input) {
	for (int j = 0; j < p_input->size(); j++) {
		prepareInputSTS(p_index, p_input->at(j));
		p_STS_learner->train(_STSinput[p_index]);
	}
	p_STS_learner->reset_context();
}
