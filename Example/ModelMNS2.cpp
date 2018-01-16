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

using namespace MNS;

ModelMNS2::ModelMNS2() {
    _F5 = nullptr;
    _STS = nullptr;
    _PF = nullptr;
}

ModelMNS2::~ModelMNS2() {
    delete _F5;
    delete _STS;
    delete _PF;
}

void ModelMNS2::init() {
    _data.loadData("../data/Trajectories.3.vd", "../data/Trajectories.3.md");

    _F5 = new MSOM(_sizeF5input + _sizePF * _sizePF, _sizeF5, _sizeF5, NeuralGroup::KEXPONENTIAL, 0.3, 0.5);
    _STS = new MSOM(_sizeSTSinput + _sizePF * _sizePF, _sizeSTS, _sizeSTS, NeuralGroup::KEXPONENTIAL, 0.3, 0.7);
    _PF = new SOM(_sizeF5*_sizeF5 + _sizeSTS * _sizeSTS, _sizePF, _sizePF, NeuralGroup::KEXPONENTIAL);

	_F5input = Tensor::Zero({ _sizeF5input + _sizePF * _sizePF });
	_STSinput = Tensor::Zero({ _sizeSTSinput + _sizePF * _sizePF });
	_PFinput = Tensor::Zero({ _sizeF5 *_sizeF5 + _sizeSTS * _sizeSTS });

	for(int i = 0; i < _sizeF5input + _sizePF * _sizePF; i++) {
		if (i < _sizeF5input) {
			_f5_mask[i] = 1;
		}
		else {
			_f5_mask[i] = 0;
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
}

void ModelMNS2::run(int p_epochs) {
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
			preactivateF5(train_data->at(i)->getMotorData());
			
            for(int p = 0; p < PERSPS; p++) {
				preactivateSTS(train_data->at(i)->getVisualData(p));

                prepareInputPF();
                PF_learner.train(&_PFinput);
                _PF->activate(&_PFinput);

				trainSTS(STS_learner, train_data->at(i)->getVisualData(p));
				trainF5(F5_learner, train_data->at(i)->getMotorData());
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

void ModelMNS2::prepareInputSTS(Tensor *p_input) {
    _STSinput = Tensor::Concat(*p_input, *_PF->get_output());
}

void ModelMNS2::prepareInputF5(Tensor *p_input) {
    _F5input = Tensor::Concat(*p_input, *_PF->get_output());
}

void ModelMNS2::prepareInputPF() {
    _PFinput = Tensor::Concat(*_STS->get_output(), *_F5->get_output());
}

void ModelMNS2::save_results(const string p_filename, const int p_dim_x, const int p_dim_y, double** p_data, const int p_category) const {
	ofstream file(p_filename);

	if (file.is_open()) {
		file << p_dim_x << "," << p_dim_y << endl;
		for (int i = 0; i < p_dim_x * p_dim_y; i++) {
			for (int j = 0; j < p_category; j++) {
				if (j == p_category - 1) {
					file << p_data[i][j];
				}
				else {
					file << p_data[i][j] << ",";
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

    double winRateF5_Motor[_sizeF5 * _sizeF5][GRASPS];
	double winRateF5_Visual[_sizeF5 * _sizeF5][PERSPS];
    double winRateSTS_Visual[_sizeSTS * _sizeSTS][PERSPS];
    double winRateSTS_Motor[_sizeSTS * _sizeSTS][GRASPS];
    double winRatePF_Visual[_sizePF * _sizePF][PERSPS];
    double winRatePF_Motor[_sizePF * _sizePF][GRASPS];

    for(int i = 0; i < _sizeF5 * _sizeF5; i++) {
        for(int j = 0; j < 3; j++) {
            winRateF5_Motor[i][j] = 0;
        }
		for (int j = 0; j < PERSPS; j++) {
			winRateF5_Visual[i][j] = 0;
		}
    }

    for(int i = 0; i < _sizeSTS * _sizeSTS; i++) {
        for(int j = 0; j < PERSPS; j++) {
            winRateSTS_Visual[i][j] = 0;
        }
        for(int j = 0; j < GRASPS; j++) {
            winRateSTS_Motor[i][j] = 0;
        }
    }

    for(int i = 0; i < _sizePF * _sizePF; i++) {
        for(int j = 0; j < PERSPS; j++) {
            winRatePF_Visual[i][j] = 0;
        }
        for(int j = 0; j < GRASPS; j++) {
            winRatePF_Motor[i][j] = 0;
        }
    }

	for (int i = 0; i < trainData->size(); i++) {
		preactivateF5(trainData->at(i)->getMotorData());

		for (int p = 0; p < PERSPS; p++) {
			preactivateSTS(trainData->at(i)->getVisualData(p));

			prepareInputPF();
			_PF->activate(&_PFinput);

			for (int n = 0; n < _PF->get_lattice()->getDim(); n++) {
				winRatePF_Visual[n][p] += _PF->get_output()->at(n);
				winRatePF_Motor[n][trainData->at(i)->getGrasp() - 1] += _PF->get_output()->at(n);
			}

			for (int j = 0; j < trainData->at(i)->getVisualData(p)->size(); j++) {
				prepareInputSTS(trainData->at(i)->getVisualData(p)->at(j));
				_STS->activate(&_STSinput);
			}
			for (int n = 0; n < _STS->get_lattice()->getDim(); n++) {
				winRateSTS_Visual[n][p] += _STS->get_output()->at(n);
				winRateSTS_Motor[n][trainData->at(i)->getGrasp() - 1] += _STS->get_output()->at(n);
			}
			_STS->reset_context();

			for (int j = 0; j < trainData->at(i)->getMotorData()->size(); j++) {
				prepareInputF5(trainData->at(i)->getMotorData()->at(j));
				_F5->activate(&_F5input);
			}
			for (int n = 0; n < _F5->get_lattice()->getDim(); n++) {
				winRateF5_Visual[n][p] += _F5->get_output()->at(n);
				winRateF5_Motor[n][trainData->at(i)->getGrasp() - 1] += _F5->get_output()->at(n);
			}
			_F5->reset_context();
		}
	}

	//save_results(timestamp + "_F5.mot", _sizeF5, _sizeF5, reinterpret_cast<double**>(winRateF5_Motor), GRASPS);

    ofstream motFile(timestamp + "_F5.mot");

    if (motFile.is_open()) {
        motFile << _sizeF5 << "," << _sizeF5 << endl;
        for (int i = 0; i < _sizeF5 * _sizeF5; i++) {
            for (int j = 0; j < GRASPS; j++) {
                if (j == GRASPS - 1) {
                    motFile << winRateF5_Motor[i][j];
                }
                else {
                    motFile << winRateF5_Motor[i][j] << ",";
                }
            }
            if (i < _sizeF5 * _sizeF5 - 1) motFile << endl;
        }
    }

    motFile.close();

	ofstream visFile(timestamp + "_F5.vis");

	if (visFile.is_open()) {
		visFile << _sizeF5 << "," << _sizeF5 << endl;
		for (int i = 0; i < _sizeF5 * _sizeF5; i++) {
			for (int j = 0; j < PERSPS; j++) {
				if (j == PERSPS - 1) {
					visFile << winRateF5_Visual[i][j];
				}
				else {
					visFile << winRateF5_Visual[i][j] << ",";
				}
			}
			if (i < _sizeF5 * _sizeF5 - 1) visFile << endl;
		}
	}

	visFile.close();

    ofstream STSvisFile(timestamp + "_STS.vis");

    if (STSvisFile.is_open()) {
        STSvisFile << _sizeSTS << "," << _sizeSTS << endl;
        for (int i = 0; i < _sizeSTS * _sizeSTS; i++) {
            for (int j = 0; j < PERSPS; j++) {
                if (j == PERSPS - 1) {
                    STSvisFile << winRateSTS_Visual[i][j];
                }
                else {
                    STSvisFile << winRateSTS_Visual[i][j] << ",";
                }
            }
            if (i < _sizeSTS * _sizeSTS - 1) STSvisFile << endl;
        }
    }

    STSvisFile.close();

    ofstream STSmotFile(timestamp + "_STS.mot");

    if (STSmotFile.is_open()) {
        STSmotFile << _sizeSTS << "," << _sizeSTS << endl;
        for (int i = 0; i < _sizeSTS * _sizeSTS; i++) {
            for (int j = 0; j < GRASPS; j++) {
                if (j == GRASPS - 1) {
                    STSmotFile << winRateSTS_Motor[i][j];
                }
                else {
                    STSmotFile << winRateSTS_Motor[i][j] << ",";
                }
            }
            if (i < _sizeSTS * _sizeSTS - 1) STSmotFile << endl;
        }
    }

    STSmotFile.close();

    ofstream PFvisFile(timestamp + "_PF.vis");

    if (PFvisFile.is_open()) {
        PFvisFile << _sizePF << "," << _sizePF << endl;
        for (int i = 0; i < _sizePF * _sizePF; i++) {
            for (int j = 0; j < PERSPS; j++) {
                if (j == PERSPS - 1) {
                    PFvisFile << winRatePF_Visual[i][j];
                }
                else {
                    PFvisFile << winRatePF_Visual[i][j] << ",";
                }
            }
            if (i < _sizePF * _sizePF - 1) PFvisFile << endl;
        }
    }

    PFvisFile.close();

    ofstream PFmotFile(timestamp + "_PF.mot");

    if (PFmotFile.is_open()) {
        PFmotFile << _sizePF << "," << _sizePF << endl;
        for (int i = 0; i < _sizePF * _sizePF; i++) {
            for (int j = 0; j < GRASPS; j++) {
                if (j == GRASPS - 1) {
                    PFmotFile << winRatePF_Motor[i][j];
                }
                else {
                    PFmotFile << winRatePF_Motor[i][j] << ",";
                }
            }
            if (i < _sizePF * _sizePF - 1) PFmotFile << endl;
        }
    }

    PFmotFile.close();
}

void ModelMNS2::testFinalWinners() {
    const string timestamp = to_string(time(nullptr));

    vector<Sequence*>* trainData = _data.permute();
    int winRateF5_Motor[_sizeF5 * _sizeF5][GRASPS];
	int winRateF5_Visual[_sizeF5 * _sizeF5][PERSPS];
    int winRateSTS_Visual[_sizeSTS * _sizeSTS][PERSPS];
    int winRateSTS_Motor[_sizeSTS * _sizeSTS][GRASPS];
    int winRatePF_Visual[_sizePF * _sizePF][PERSPS];
    int winRatePF_Motor[_sizePF * _sizePF][GRASPS];
    

    for(int i = 0; i < _sizeF5 * _sizeF5; i++) {
        for(int j = 0; j < GRASPS; j++) {
            winRateF5_Motor[i][j] = 0;
        }
		for (int j = 0; j < PERSPS; j++) {
			winRateF5_Visual[i][j] = 0;
		}
    }

    for(int i = 0; i < _sizeSTS * _sizeSTS; i++) {
        for(int j = 0; j < PERSPS; j++) {
            winRateSTS_Visual[i][j] = 0;
        }
        for(int j = 0; j < GRASPS; j++) {
            winRateSTS_Motor[i][j] = 0;
        }
    }

    for(int i = 0; i < _sizePF * _sizePF; i++) {
        for(int j = 0; j < PERSPS; j++) {
            winRatePF_Visual[i][j] = 0;
        }
        for(int j = 0; j < GRASPS; j++) {
            winRatePF_Motor[i][j] = 0;
        }
    }

	for (int i = 0; i < trainData->size(); i++) {
		preactivateF5(trainData->at(i)->getMotorData());

		for (int p = 0; p < PERSPS; p++) {
			preactivateSTS(trainData->at(i)->getVisualData(p));

			prepareInputPF();
			_PF->activate(&_PFinput);
			winRatePF_Visual[_PF->get_winner()][p]++;
			winRatePF_Motor[_PF->get_winner()][trainData->at(i)->getGrasp() - 1]++;

			for (int j = 0; j < trainData->at(i)->getVisualData(p)->size(); j++) {
				prepareInputSTS(trainData->at(i)->getVisualData(p)->at(j));
				_STS->activate(&_STSinput);
			}
			winRateSTS_Visual[_STS->get_winner()][p]++;
			winRateSTS_Motor[_STS->get_winner()][trainData->at(i)->getGrasp() - 1]++;
			_STS->reset_context();

			for (int j = 0; j < trainData->at(i)->getMotorData()->size(); j++) {
				prepareInputF5(trainData->at(i)->getMotorData()->at(j));
				_F5->activate(&_F5input);
			}
			winRateF5_Visual[_F5->get_winner()][p]++;
			winRateF5_Motor[_F5->get_winner()][trainData->at(i)->getGrasp() - 1]++;
			_F5->reset_context();
		}
	}

    ofstream motFile(timestamp + "_F5.mot");

    if (motFile.is_open()) {
        motFile << _sizeF5 << "," << _sizeF5 << endl;
        for (int i = 0; i < _sizeF5 * _sizeF5; i++) {
            for (int j = 0; j < GRASPS; j++) {
                if (j == GRASPS - 1) {
                    motFile << winRateF5_Motor[i][j];
                }
                else {
                    motFile << winRateF5_Motor[i][j] << ",";
                }
            }
            if (i < _sizeF5 * _sizeF5 - 1) motFile << endl;
        }
    }

    motFile.close();

	ofstream visFile(timestamp + "_F5.vis");

	if (visFile.is_open()) {
		visFile << _sizeF5 << "," << _sizeF5 << endl;
		for (int i = 0; i < _sizeF5 * _sizeF5; i++) {
			for (int j = 0; j < PERSPS; j++) {
				if (j == PERSPS - 1) {
					visFile << winRateF5_Visual[i][j];
				}
				else {
					visFile << winRateF5_Visual[i][j] << ",";
				}
			}
			if (i < _sizeF5 * _sizeF5 - 1) visFile << endl;
		}
	}

	visFile.close();

    ofstream STSvisFile(timestamp + "_STS.vis");

    if (STSvisFile.is_open()) {
        STSvisFile << _sizeSTS << "," << _sizeSTS << endl;
        for (int i = 0; i < _sizeSTS * _sizeSTS; i++) {
            for (int j = 0; j < PERSPS; j++) {
                if (j == PERSPS - 1) {
                    STSvisFile << winRateSTS_Visual[i][j];
                }
                else {
                    STSvisFile << winRateSTS_Visual[i][j] << ",";
                }
            }
            if (i < _sizeSTS * _sizeSTS - 1) STSvisFile << endl;
        }
    }

    STSvisFile.close();

    ofstream STSmotFile(timestamp + "_STS.mot");

    if (STSmotFile.is_open()) {
        STSmotFile << _sizeSTS << "," << _sizeSTS << endl;
        for (int i = 0; i < _sizeSTS * _sizeSTS; i++) {
            for (int j = 0; j < GRASPS; j++) {
                if (j == GRASPS - 1) {
                    STSmotFile << winRateSTS_Motor[i][j];
                }
                else {
                    STSmotFile << winRateSTS_Motor[i][j] << ",";
                }
            }
            if (i < _sizeSTS * _sizeSTS - 1) STSmotFile << endl;
        }
    }

    STSmotFile.close();

    ofstream PFvisFile(timestamp + "_PF.vis");

    if (PFvisFile.is_open()) {
        PFvisFile << _sizePF << "," << _sizePF << endl;
        for (int i = 0; i < _sizePF * _sizePF; i++) {
            for (int j = 0; j < PERSPS; j++) {
                if (j == PERSPS - 1) {
                    PFvisFile << winRatePF_Visual[i][j];
                }
                else {
                    PFvisFile << winRatePF_Visual[i][j] << ",";
                }
            }
            if (i < _sizePF * _sizePF - 1) PFvisFile << endl;
        }
    }

    PFvisFile.close();

    ofstream PFmotFile(timestamp + "_PF.mot");

    if (PFmotFile.is_open()) {
        PFmotFile << _sizePF << "," << _sizePF << endl;
        for (int i = 0; i < _sizePF * _sizePF; i++) {
            for (int j = 0; j < GRASPS; j++) {
                if (j == GRASPS - 1) {
                    PFmotFile << winRatePF_Motor[i][j];
                }
                else {
                    PFmotFile << winRatePF_Motor[i][j] << ",";
                }
            }
            if (i < _sizePF * _sizePF - 1) PFmotFile << endl;
        }
    }

    PFmotFile.close();
}

void ModelMNS2::preactivateF5(vector<Tensor*>* p_input) {
	_F5->set_input_mask(_f5_mask);

	for (int j = 0; j < p_input->size(); j++) {
		prepareInputF5(p_input->at(j));
		_F5->activate(&_F5input);
	}
	_F5->reset_context();

	_F5->set_input_mask(nullptr);
}

void ModelMNS2::preactivateSTS(vector<Tensor*>* p_input) {
	_STS->set_input_mask(_sts_mask);

	for (int j = 0; j < p_input->size(); j++) {
		prepareInputSTS(p_input->at(j));
		_STS->activate(&_STSinput);
	}
	_STS->reset_context();

	_STS->set_input_mask(nullptr);
}

void ModelMNS2::trainF5(MSOM_learning& p_F5_learner, vector<Tensor*>* p_input) {
	for (int j = 0; j < p_input->size(); j++) {
		prepareInputF5(p_input->at(j));
		p_F5_learner.train(&_F5input);
	}
	_F5->reset_context();
}

void ModelMNS2::trainSTS(MSOM_learning& p_STS_learner, vector<Tensor*>* p_input) {
	for (int j = 0; j < p_input->size(); j++) {
		prepareInputSTS(p_input->at(j));
		p_STS_learner.train(&_STSinput);
	}
	_STS->reset_context();
}
