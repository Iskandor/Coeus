//
// Created by user on 14. 12. 2017.
//

#include <fstream>
#include <chrono>
#include "ModelMNS2.h"
#include "MSOM_learning.h"
#include "SOM_learning.h"
#include <iostream>
#include <string>
#include "IOUtils.h"
#include "Config.h"
#include <concrtrm.h>

using namespace MNS;
using namespace Concurrency;

ModelMNS2::ModelMNS2() {
    _F5 = nullptr;
    _STS = nullptr;
    _PF = nullptr;
}

ModelMNS2::~ModelMNS2() {
    delete _F5;
    delete _STS;
    delete _PF;

	vector<Sequence*>* trainData = _data.permute();

	for (int i = 0; i < trainData->size(); i++) {
		delete _F5input[i];
	}
	for (int i = 0; i < trainData->size() * PERSPS; i++) {
		delete _STSinput[i];
		delete _PFinput[i];
	}

	delete _F5input;
	delete _STSinput;
	delete _PFinput;
}

void ModelMNS2::init() {
    _data.loadData("../data/Trajectories.3.vd", "../data/Trajectories.3.md");

    _F5 = new MSOM(_sizeF5input + _sizePF * _sizePF, _sizeF5, _sizeF5, NeuralGroup::KEXPONENTIAL, 0.3, 0.5);
    _STS = new MSOM(_sizeSTSinput + _sizePF * _sizePF, _sizeSTS, _sizeSTS, NeuralGroup::KEXPONENTIAL, 0.3, 0.7);
    _PF = new SOM(_sizeF5*_sizeF5 + _sizeSTS * _sizeSTS, _sizePF, _sizePF, NeuralGroup::KEXPONENTIAL);

	for(int i = 0; i < _sizeF5input + _sizePF * _sizePF; i++) {
		if (i < _sizeF5input) {
			_f5_mask_pre[i] = 1;
			_f5_mask_post[i] = 0;
		}
		else {
			_f5_mask_pre[i] = 0;
			_f5_mask_post[i] = 1;
		}
	}

	for (int i = 0; i < _sizeSTSinput + _sizePF * _sizePF; i++) {
		if (i < _sizeSTSinput) {
			_sts_mask[i] = 1;
		}
		else {
			_sts_mask[i] = 0;
		}
	}

	for(int i = 0; i < _sizeSTS * _sizeSTS + _sizeF5 *_sizeF5; i++) {
		if (i < _sizeSTS * _sizeSTS) {
			_pf_mask[i] = 1;
		}
		else {
			_pf_mask[i] = 0;
		}
	}

	vector<Sequence*>* trainData = _data.permute();

	_F5input = new Tensor*[trainData->size()];
	_STSinput = new Tensor*[trainData->size() * PERSPS];
	_PFinput = new Tensor*[trainData->size() * PERSPS];

	for (int i = 0; i < trainData->size(); i++) {
		_F5input[i] = new Tensor({ _sizeF5input + _sizeSTS * _sizeSTS }, Tensor::ZERO);		
	}
	for (int i = 0; i < trainData->size() * PERSPS; i++) {
		_STSinput[i] = new Tensor({ _sizeSTSinput + _sizeF5 * _sizeF5 }, Tensor::ZERO);
		_PFinput[i] = new Tensor({ _sizeSTS * _sizeSTS + _sizeF5 *_sizeF5 }, Tensor::ZERO);
	}

	
}

void ModelMNS2::run(int p_epochs) {
	cout << "Epochs: " << p_epochs << endl;
	cout << "Settling: " << Config::instance().settling << endl;
	cout << "CPUs: " << GetProcessorCount() << endl;
	cout << "Initializing learning module...";

	SOM_analyzer F5_analyzer;
	SOM_analyzer STS_analyzer;
	SOM_analyzer PF_analyzer;

	MSOM_params F5_params(_F5);
	F5_params.init_training(Config::instance().f5_config.gamma1, Config::instance().f5_config.gamma2, p_epochs);

	MSOM_params STS_params(_STS);
	STS_params.init_training(Config::instance().sts_config.gamma1, Config::instance().sts_config.gamma2, p_epochs);

	SOM_params PF_params(_PF);
	PF_params.init_training(Config::instance().pf_config.alpha, p_epochs);

	MSOM_learning F5_learner(_F5, &F5_params, &F5_analyzer);
	MSOM_learning STS_learner(_STS, &STS_params, &STS_analyzer);
	SOM_learning PF_learner(_PF, &PF_params, &PF_analyzer);

	vector<Sequence*>* trainData = _data.permute();

	vector<MSOM_learning*> F5_thread(trainData->size());

	for (int i = 0; i < trainData->size(); i++) {
		F5_thread[i] = new MSOM_learning(_F5->clone(), &F5_params, &F5_analyzer);
	}

	vector<MSOM_learning*> STS_thread(trainData->size() * PERSPS);

	for (int i = 0; i < trainData->size() * PERSPS; i++) {
		STS_thread[i] = new MSOM_learning(_STS->clone(), &STS_params, &STS_analyzer);
	}

	cout << "done" << endl;

	/*
	MSOM_learning F5_learner(_F5);
	F5_learner.init_training(0.01, 0.01, p_epochs);
	MSOM_learning STS_learner(_STS);
    STS_learner.init_training(0.1, 0.1, p_epochs);
	SOM_learning PF_learner(_PF);
    PF_learner.init_training(0.1, p_epochs);

	for(int t = 0; t < p_epochs; t++) {
        cout << "Epoch " << t << endl;
        vector<Sequence*> * train_data = _data.permute();

        auto start = chrono::system_clock::now();
        for(int i = 0; i < train_data->size(); i++) {
			_F5->set_input_mask(_f5_mask_pre);
			activateF5(train_data->at(i)->getMotorData());
			_F5->set_input_mask(nullptr);
			
            for(int p = 0; p < PERSPS; p++) {
				_STS->set_input_mask(_sts_mask);
				activateSTS(train_data->at(i)->getVisualData(p));
				_STS->set_input_mask(nullptr);

                prepareInputPF();
                PF_learner.train(&_PFinput);
                _PF->activate(&_PFinput);

				trainSTS(STS_learner, train_data->at(i)->getVisualData(p));
				trainF5(F5_learner, train_data->at(i)->getMotorData());

				for (int s = 0; s < 40; s++) {
					activateSTS(train_data->at(i)->getVisualData(p));
					activateF5(train_data->at(i)->getMotorData());

					prepareInputPF();
					PF_learner.train(&_PFinput);
					_PF->activate(&_PFinput);

					trainSTS(STS_learner, train_data->at(i)->getVisualData(p));
					trainF5(F5_learner, train_data->at(i)->getMotorData());
				}
            }
        }

        auto end = chrono::system_clock::now();
        chrono::duration<double> elapsed_seconds = end-start;
        cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
        cout << " F5 qError: " << F5_learner.analyzer()->q_error() << " WD: " << F5_learner.analyzer()->winner_diff() << endl;
        cout << "STS qError: " << STS_learner.analyzer()->q_error() << " WD: " << STS_learner.analyzer()->winner_diff() << endl;
        cout << " PF qError: " << PF_learner.analyzer()->q_error() << " WD: " << PF_learner.analyzer()->winner_diff() << endl;
		F5_learner.param_decay();
		STS_learner.param_decay();
		PF_learner.param_decay();		
    }
	*/
}

void ModelMNS2::save() {
    const string timestamp = to_string(time(nullptr));

    IOUtils::save_network(timestamp + "_F5.json", _F5);
	IOUtils::save_network(timestamp + "_STS.json", _STS);
	IOUtils::save_network(timestamp + "_PF.json", _PF);
}

void ModelMNS2::load(const string p_timestamp) {
    _F5 = static_cast<MSOM*>(IOUtils::load_network("C:\\GIT\\Coeus\\x64\\Debug\\" + p_timestamp + "_F5.json"));
    _STS = static_cast<MSOM*>(IOUtils::load_network("C:\\GIT\\Coeus\\x64\\Debug\\" + p_timestamp + "_STS.json"));
    _PF = static_cast<SOM*>(IOUtils::load_network("C:\\GIT\\Coeus\\x64\\Debug\\" + p_timestamp + "_PF.json"));
}

void ModelMNS2::prepareInputSTS(int p_index, Tensor *p_input) {
	Tensor::Concat(_STSinput[p_index], p_input, _PF->get_output());
}

void ModelMNS2::prepareInputF5(int p_index, Tensor *p_input) {
	Tensor::Concat(_F5input[p_index], p_input, _PF->get_output());
}

void ModelMNS2::prepareInputPF(int p_index) {
	Tensor::Concat(_PFinput[p_index], _STS->get_output(), _F5->get_output());

}

void ModelMNS2::save_results(const string p_filename, const int p_dim_x, const int p_dim_y, double* p_data, const int p_category) const {
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

void ModelMNS2::testDistance() {
    const string timestamp = to_string(time(nullptr));

    vector<Sequence*>* trainData = _data.permute();

    double* winRateF5_Motor = static_cast<double*>(calloc(_sizeF5 * _sizeF5 * GRASPS, sizeof(double)));
	double* winRateF5_Visual = static_cast<double*>(calloc(_sizeF5 * _sizeF5 * PERSPS, sizeof(double)));
    double* winRateSTS_Visual = static_cast<double*>(calloc(_sizeSTS * _sizeSTS * PERSPS, sizeof(double)));
    double* winRateSTS_Motor = static_cast<double*>(calloc(_sizeSTS * _sizeSTS * GRASPS, sizeof(double)));
    double* winRatePF_Visual = static_cast<double*>(calloc(_sizePF * _sizePF * PERSPS, sizeof(double)));
    double* winRatePF_Motor = static_cast<double*>(calloc(_sizePF * _sizePF * GRASPS, sizeof(double)));

	for (int i = 0; i < trainData->size(); i++) {
		_F5->set_input_mask(_f5_mask_pre);
		activateF5(i, _F5, trainData->at(i)->getMotorData());
		_F5->set_input_mask(nullptr);

		for (int p = 0; p < PERSPS; p++) {
			_STS->set_input_mask(_sts_mask);
			activateSTS(i, _STS, trainData->at(i)->getVisualData(p));
			_STS->set_input_mask(nullptr);

			for(int s = 0; s < 10; s++) {
				activatePF(i, _PF);

				activateSTS(i, _STS, trainData->at(i)->getVisualData(p));

				activateF5(i, _F5, trainData->at(i)->getMotorData());
			}

			for (int n = 0; n < _PF->get_lattice()->getDim(); n++) {
				winRatePF_Visual[n * PERSPS + p] += _PF->get_output()->at(n);
				winRatePF_Motor[n * GRASPS + trainData->at(i)->getGrasp() - 1] += _PF->get_output()->at(n);
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
	save_results(timestamp + "_PF.mot", _sizePF, _sizePF, winRatePF_Motor, GRASPS);
	save_results(timestamp + "_PF.vis", _sizePF, _sizePF, winRatePF_Visual, PERSPS);
}

void ModelMNS2::testFinalWinners() {
    const string timestamp = to_string(time(nullptr));

    vector<Sequence*>* trainData = _data.permute();

	double* winRateF5_Motor = static_cast<double*>(calloc(_sizeF5 * _sizeF5 * GRASPS, sizeof(double)));
	double* winRateF5_Visual = static_cast<double*>(calloc(_sizeF5 * _sizeF5 * PERSPS, sizeof(double)));
	double* winRateSTS_Visual = static_cast<double*>(calloc(_sizeSTS * _sizeSTS * PERSPS, sizeof(double)));
	double* winRateSTS_Motor = static_cast<double*>(calloc(_sizeSTS * _sizeSTS * GRASPS, sizeof(double)));
	double* winRatePF_Visual = static_cast<double*>(calloc(_sizePF * _sizePF * PERSPS, sizeof(double)));
	double* winRatePF_Motor = static_cast<double*>(calloc(_sizePF * _sizePF * GRASPS, sizeof(double)));
 
	for (int i = 0; i < trainData->size(); i++) {
		_F5->set_input_mask(_f5_mask_pre);
		activateF5(i, _F5, trainData->at(i)->getMotorData());
		_F5->set_input_mask(nullptr);

		for (int p = 0; p < PERSPS; p++) {
			_STS->set_input_mask(_sts_mask);
			activateSTS(i, _STS, trainData->at(i)->getVisualData(p));
			_STS->set_input_mask(nullptr);

			for(int s = 0; s < 10; s++) {
				activatePF(i, _PF);
				activateSTS(i, _STS, trainData->at(i)->getVisualData(p));
				activateF5(i, _F5, trainData->at(i)->getMotorData());
			}

			winRatePF_Visual[_PF->get_winner() * PERSPS + p]++;
			winRatePF_Motor[_PF->get_winner() * GRASPS + trainData->at(i)->getGrasp() - 1]++;

			winRateSTS_Visual[_STS->get_winner() * PERSPS + p]++;
			winRateSTS_Motor[_STS->get_winner() * GRASPS + trainData->at(i)->getGrasp() - 1]++;

			winRateF5_Visual[_F5->get_winner() * PERSPS + p]++;
			winRateF5_Motor[_F5->get_winner() * GRASPS + trainData->at(i)->getGrasp() - 1]++;
		}
	}

	save_results(timestamp + "_F5.mot", _sizeF5, _sizeF5, winRateF5_Motor, GRASPS);
	save_results(timestamp + "_F5.vis", _sizeF5, _sizeF5, winRateF5_Visual, PERSPS);
	save_results(timestamp + "_STS.mot", _sizeSTS, _sizeSTS, winRateSTS_Motor, GRASPS);
	save_results(timestamp + "_STS.vis", _sizeSTS, _sizeSTS, winRateSTS_Visual, PERSPS);
	save_results(timestamp + "_PF.mot", _sizePF, _sizePF, winRatePF_Motor, GRASPS);
	save_results(timestamp + "_PF.vis", _sizePF, _sizePF, winRatePF_Visual, PERSPS);
}

void ModelMNS2::testMirror() {
	const string timestamp = to_string(time(nullptr));

	vector<Sequence*>* trainData = _data.permute();

	double* winRateF5_Motor = static_cast<double*>(calloc(_sizeF5 * _sizeF5 * GRASPS, sizeof(double)));
	double* winRateF5_Visual = static_cast<double*>(calloc(_sizeF5 * _sizeF5 * PERSPS, sizeof(double)));
	double* winRateSTS_Visual = static_cast<double*>(calloc(_sizeSTS * _sizeSTS * PERSPS, sizeof(double)));
	double* winRateSTS_Motor = static_cast<double*>(calloc(_sizeSTS * _sizeSTS * GRASPS, sizeof(double)));
	double* winRatePF_Visual = static_cast<double*>(calloc(_sizePF * _sizePF * PERSPS, sizeof(double)));
	double* winRatePF_Motor = static_cast<double*>(calloc(_sizePF * _sizePF * GRASPS, sizeof(double)));

	for (int i = 0; i < 1; i++) {
		for (int p = 0; p < PERSPS; p++) {
			_STS->set_input_mask(_sts_mask);
			activateSTS(i, _STS, trainData->at(i)->getVisualData(p));
			_PF->set_input_mask(_pf_mask);
			activatePF(i, _PF);
			_F5->set_input_mask(_f5_mask_post);
			activateF5(i, _F5, trainData->at(i)->getMotorData());
			_PF->set_input_mask(nullptr);
			activatePF(i, _PF);
			_STS->set_input_mask(nullptr);
			activateSTS(i, _STS, trainData->at(i)->getVisualData(p));

			for(int s = 0; s < 10; s++) {
				_PF->set_input_mask(nullptr);
				activatePF(i, _PF);
				_F5->set_input_mask(_f5_mask_post);
				activateF5(i, _F5, trainData->at(i)->getMotorData());
				_STS->set_input_mask(nullptr);
				activateSTS(i, _STS, trainData->at(i)->getVisualData(p));
			}	

			for (int n = 0; n < _STS->get_lattice()->getDim(); n++) {
				winRateSTS_Visual[n * PERSPS + p] += _STS->get_output()->at(n);
				winRateSTS_Motor[n * GRASPS + trainData->at(i)->getGrasp() - 1] += _STS->get_output()->at(n);
			}		
			
			for (int n = 0; n < _F5->get_lattice()->getDim(); n++) {
				winRateF5_Visual[n * PERSPS + p] += _F5->get_output()->at(n);
				winRateF5_Motor[n * GRASPS + trainData->at(i)->getGrasp() - 1] += _F5->get_output()->at(n);
			}

			for (int n = 0; n < _PF->get_lattice()->getDim(); n++) {
				winRatePF_Visual[n * PERSPS + p] += _PF->get_output()->at(n);
				winRatePF_Motor[n * GRASPS + trainData->at(i)->getGrasp() - 1] += _PF->get_output()->at(n);
			}		
		}
	}

	save_results(timestamp + "_F5.mot", _sizeF5, _sizeF5, winRateF5_Motor, GRASPS);
	save_results(timestamp + "_F5.vis", _sizeF5, _sizeF5, winRateF5_Visual, PERSPS);
	save_results(timestamp + "_STS.mot", _sizeSTS, _sizeSTS, winRateSTS_Motor, GRASPS);
	save_results(timestamp + "_STS.vis", _sizeSTS, _sizeSTS, winRateSTS_Visual, PERSPS);
	save_results(timestamp + "_PF.mot", _sizePF, _sizePF, winRatePF_Motor, GRASPS);
	save_results(timestamp + "_PF.vis", _sizePF, _sizePF, winRatePF_Visual, PERSPS);
}

void ModelMNS2::activatePF(int p_index, SOM* p_som) {
	prepareInputPF(p_index);
	p_som->activate(_PFinput[p_index]);
}

void MNS::ModelMNS2::activateF5(int p_index, MSOM* p_msom, vector<Tensor*>* p_input)
{
	for (int j = 0; j < p_input->size(); j++) {
		prepareInputF5(p_index, p_input->at(j));
		p_msom->activate(_F5input[p_index]);
	}
	p_msom->reset_context();

}

void MNS::ModelMNS2::activateSTS(int p_index, MSOM* p_msom, vector<Tensor*>* p_input)
{
	for (int j = 0; j < p_input->size(); j++) {
		prepareInputSTS(p_index, p_input->at(j));
		p_msom->activate(_STSinput[p_index]);
	}
	p_msom->reset_context();
}

void ModelMNS2::trainF5(int p_index, MSOM_learning* p_F5_learner, vector<Tensor*>* p_input) {
	for (int j = 0; j < p_input->size(); j++) {
		prepareInputF5(p_index, p_input->at(j));
		p_F5_learner->train(_F5input[p_index]);
	}
	p_F5_learner->reset_context();
}

void ModelMNS2::trainSTS(int p_index, MSOM_learning* p_STS_learner, vector<Tensor*>* p_input) {
	for (int j = 0; j < p_input->size(); j++) {
		prepareInputSTS(p_index, p_input->at(j));
		p_STS_learner->train(_STSinput[p_index]);
	}
	p_STS_learner->reset_context();
}
