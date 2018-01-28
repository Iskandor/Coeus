//
// Created by user on 5. 11. 2017.
//

#include <fstream>
#include <chrono>
#include "ModelMNS.h"
#include <iostream>
#include <string>
#include "MSOM_learning.h"
#include "IOUtils.h"

using namespace MNS;
using namespace std;

ModelMNS::ModelMNS() {
    _msomMotor = nullptr;
    _msomVisual = nullptr;
}

ModelMNS::~ModelMNS() {
    delete _msomMotor;
    delete _msomVisual;
}

void ModelMNS::init() {
    _data.loadData("../data/Trajectories.3.vd", "../data/Trajectories.3.md");

    _msomMotor = new MSOM(16, _sizePMC, _sizePMC, NeuralGroup::EXPONENTIAL, 0.3, 0.5);
    _msomVisual = new MSOM(40, _sizeSTSp, _sizeSTSp, NeuralGroup::EXPONENTIAL, 0.3, 0.7);

}

void ModelMNS::run(int p_epochs) {
	MSOM_learning F5_learner(_msomMotor);
	F5_learner.init_training(0.001, 0.001, p_epochs);
	MSOM_learning STS_learner(_msomVisual);
	STS_learner.init_training(0.001, 0.001, p_epochs);

    vector<Sequence*>* trainData = nullptr;

    for(int t = 0; t < p_epochs; t++) {
        cout << "Epoch " << t << endl;
        trainData = _data.permute();

        auto start = chrono::system_clock::now();
        for(int i = 0; i < trainData->size(); i++) {
            for(int j = 0; j < trainData->at(i)->getMotorData()->size(); j++) {
				F5_learner.train(trainData->at(i)->getMotorData()->at(j));
            }
            _msomMotor->reset_context();
            for(int p = 0; p < PERSPS; p++) {
                for(int j = 0; j < trainData->at(i)->getVisualData(p)->size(); j++) {
					STS_learner.train(trainData->at(i)->getVisualData(p)->at(j));
                }
                _msomVisual->reset_context();
            }
        }

		F5_learner.update();
		STS_learner.update();

        auto end = chrono::system_clock::now();
        chrono::duration<double> elapsed_seconds = end-start;
        cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
        cout << " PMC qError: " << F5_learner.analyzer()->q_error() << " WD: " << F5_learner.analyzer()->winner_diff() << endl;
        cout << "STSp qError: " << STS_learner.analyzer()->q_error() << " WD: " << STS_learner.analyzer()->winner_diff() << endl;
		F5_learner.param_decay();
		STS_learner.param_decay();
    }
}


void ModelMNS::testAllWinners() {
    const string timestamp = to_string(time(nullptr));

    vector<Sequence*>* trainData = _data.permute();
    double winRatePMC_Motor[_sizePMC * _sizePMC][GRASPS];
    double winRateSTSp_Visual[_sizeSTSp * _sizeSTSp][PERSPS];
    double winRateSTSp_Motor[_sizeSTSp * _sizeSTSp][GRASPS];

    for(int i = 0; i < _sizePMC * _sizePMC; i++) {
        for(int j = 0; j < 3; j++) {
            winRatePMC_Motor[i][j] = 0;
        }
    }

    for(int i = 0; i < _sizeSTSp * _sizeSTSp; i++) {
        for(int j = 0; j < PERSPS; j++) {
            winRateSTSp_Visual[i][j] = 0;
        }
        for(int j = 0; j < GRASPS; j++) {
            winRateSTSp_Motor[i][j] = 0;
        }
    }

    for(int i = 0; i < trainData->size(); i++) {
        for(int j = 0; j < trainData->at(i)->getMotorData()->size(); j++) {
            _msomMotor->activate(trainData->at(i)->getMotorData()->at(j));
            winRatePMC_Motor[_msomMotor->get_winner()][trainData->at(i)->getGrasp() - 1]++;
        }
        _msomMotor->reset_context();
        for(int p = 0; p < PERSPS; p++) {
            for(int j = 0; j < trainData->at(i)->getVisualData(p)->size(); j++) {
                _msomVisual->activate(trainData->at(i)->getVisualData(p)->at(j));
                winRateSTSp_Visual[_msomVisual->get_winner()][p]++;
                winRateSTSp_Motor[_msomVisual->get_winner()][trainData->at(i)->getGrasp() - 1]++;
            }
            _msomVisual->reset_context();
        }
    }

    ofstream motFile(timestamp + "_pmc.mot");

    if (motFile.is_open()) {
        motFile << _sizePMC << "," << _sizePMC << endl;
        for (int i = 0; i < _sizePMC * _sizePMC; i++) {
            for (int j = 0; j < GRASPS; j++) {
                if (j == GRASPS - 1) {
                    motFile << winRatePMC_Motor[i][j];
                }
                else {
                    motFile << winRatePMC_Motor[i][j] << ",";
                }
            }
            if (i < _sizePMC * _sizePMC - 1) motFile << endl;
        }
    }

    motFile.close();

    ofstream STSvisFile(timestamp + "_stsp.vis");

    if (STSvisFile.is_open()) {
        STSvisFile << _sizeSTSp << "," << _sizeSTSp << endl;
        for (int i = 0; i < _sizeSTSp * _sizeSTSp; i++) {
            for (int j = 0; j < PERSPS; j++) {
                if (j == PERSPS - 1) {
                    STSvisFile << winRateSTSp_Visual[i][j];
                }
                else {
                    STSvisFile << winRateSTSp_Visual[i][j] << ",";
                }
            }
            if (i < _sizeSTSp * _sizeSTSp - 1) STSvisFile << endl;
        }
    }

    STSvisFile.close();

    ofstream STSmotFile(timestamp + "_stsp.mot");

    if (STSmotFile.is_open()) {
        STSmotFile << _sizeSTSp << "," << _sizeSTSp << endl;
        for (int i = 0; i < _sizeSTSp * _sizeSTSp; i++) {
            for (int j = 0; j < GRASPS; j++) {
                if (j == GRASPS - 1) {
                    STSmotFile << winRateSTSp_Motor[i][j];
                }
                else {
                    STSmotFile << winRateSTSp_Motor[i][j] << ",";
                }
            }
            if (i < _sizeSTSp * _sizeSTSp - 1) STSmotFile << endl;
        }
    }

    STSmotFile.close();
}

void ModelMNS::testFinalWinners() {
    const string timestamp = to_string(time(nullptr));

    vector<Sequence*>* trainData = _data.permute();
    double winRatePMC_Motor[_sizePMC * _sizePMC][GRASPS];
	double winRateSTSp_Visual[_sizeSTSp * _sizeSTSp][PERSPS];
	double winRateSTSp_Motor[_sizeSTSp * _sizeSTSp][GRASPS];

    for(int i = 0; i < _sizePMC * _sizePMC; i++) {
        for(int j = 0; j < 3; j++) {
            winRatePMC_Motor[i][j] = 0;
        }
    }

    for(int i = 0; i < _sizeSTSp * _sizeSTSp; i++) {
        for(int j = 0; j < PERSPS; j++) {
            winRateSTSp_Visual[i][j] = 0;
        }
        for(int j = 0; j < GRASPS; j++) {
            winRateSTSp_Motor[i][j] = 0;
        }
    }

    for(int i = 0; i < trainData->size(); i++) {
        for(int j = 0; j < trainData->at(i)->getMotorData()->size(); j++) {
            _msomMotor->activate(trainData->at(i)->getMotorData()->at(j));
        }
        winRatePMC_Motor[_msomMotor->get_winner()][trainData->at(i)->getGrasp() - 1]++;
        _msomMotor->reset_context();
        for(int p = 0; p < PERSPS; p++) {
            for(int j = 0; j < trainData->at(i)->getVisualData(p)->size(); j++) {
                _msomVisual->activate(trainData->at(i)->getVisualData(p)->at(j));
            }
            winRateSTSp_Visual[_msomVisual->get_winner()][p]++;
            winRateSTSp_Motor[_msomVisual->get_winner()][trainData->at(i)->getGrasp() - 1]++;
            _msomVisual->reset_context();
        }

    }

    ofstream motFile(timestamp + "_pmc.mot");

    if (motFile.is_open()) {
        motFile << _sizePMC << "," << _sizePMC << endl;
        for (int i = 0; i < _sizePMC * _sizePMC; i++) {
            for (int j = 0; j < GRASPS; j++) {
                if (j == GRASPS - 1) {
                    motFile << winRatePMC_Motor[i][j];
                }
                else {
                    motFile << winRatePMC_Motor[i][j] << ",";
                }
            }
            if (i < _sizePMC * _sizePMC - 1) motFile << endl;
        }
    }

    motFile.close();

    ofstream STSvisFile(timestamp + "_stsp.vis");

    if (STSvisFile.is_open()) {
        STSvisFile << _sizeSTSp << "," << _sizeSTSp << endl;
        for (int i = 0; i < _sizeSTSp * _sizeSTSp; i++) {
            for (int j = 0; j < PERSPS; j++) {
                if (j == PERSPS - 1) {
                    STSvisFile << winRateSTSp_Visual[i][j];
                }
                else {
                    STSvisFile << winRateSTSp_Visual[i][j] << ",";
                }
            }
            if (i < _sizeSTSp * _sizeSTSp - 1) STSvisFile << endl;
        }
    }

    STSvisFile.close();

    ofstream STSmotFile(timestamp + "_stsp.mot");

    if (STSmotFile.is_open()) {
        STSmotFile << _sizeSTSp << "," << _sizeSTSp << endl;
        for (int i = 0; i < _sizeSTSp * _sizeSTSp; i++) {
            for (int j = 0; j < GRASPS; j++) {
                if (j == GRASPS - 1) {
                    STSmotFile << winRateSTSp_Motor[i][j];
                }
                else {
                    STSmotFile << winRateSTSp_Motor[i][j] << ",";
                }
            }
            if (i < _sizeSTSp * _sizeSTSp - 1) STSmotFile << endl;
        }
    }

    STSmotFile.close();
}

void ModelMNS::testDistance() {
    const string timestamp = to_string(time(nullptr));

    vector<Sequence*>* trainData = _data.permute();
    double winRatePMC_Motor[_sizePMC * _sizePMC][GRASPS];
    double winRateSTSp_Visual[_sizeSTSp * _sizeSTSp][PERSPS];
    double winRateSTSp_Motor[_sizeSTSp * _sizeSTSp][GRASPS];

    for(int i = 0; i < _sizePMC * _sizePMC; i++) {
        for(int j = 0; j < 3; j++) {
            winRatePMC_Motor[i][j] = 0;
        }
    }

    for(int i = 0; i < _sizeSTSp * _sizeSTSp; i++) {
        for(int j = 0; j < PERSPS; j++) {
            winRateSTSp_Visual[i][j] = 0;
        }
        for(int j = 0; j < GRASPS; j++) {
            winRateSTSp_Motor[i][j] = 0;
        }
    }

    for(int i = 0; i < trainData->size(); i++) {
        for(int j = 0; j < trainData->at(i)->getMotorData()->size(); j++) {
            _msomMotor->activate(trainData->at(i)->getMotorData()->at(j));
        }
        for(int n = 0; n < _msomMotor->get_lattice()->getDim(); n++) {
            winRatePMC_Motor[n][trainData->at(i)->getGrasp() - 1] += _msomMotor->get_output()->at(n);
        }

        _msomMotor->reset_context();
        for(int p = 0; p < PERSPS; p++) {
            for(int j = 0; j < trainData->at(i)->getVisualData(p)->size(); j++) {
                _msomVisual->activate(trainData->at(i)->getVisualData(p)->at(j));
            }
            for(int n = 0; n < _msomVisual->get_lattice()->getDim(); n++) {
                winRateSTSp_Visual[n][p] += _msomVisual->get_output()->at(n);
                winRateSTSp_Motor[n][trainData->at(i)->getGrasp() - 1] += _msomVisual->get_output()->at(n);
            }
            _msomVisual->reset_context();
        }
    }

    ofstream motFile(timestamp + "_pmc.mot");

    if (motFile.is_open()) {
        motFile << _sizePMC << "," << _sizePMC << endl;
        for (int i = 0; i < _sizePMC * _sizePMC; i++) {
            for (int j = 0; j < GRASPS; j++) {
                if (j == GRASPS - 1) {
                    motFile << winRatePMC_Motor[i][j];
                }
                else {
                    motFile << winRatePMC_Motor[i][j] << ",";
                }
            }
            if (i < _sizePMC * _sizePMC - 1) motFile << endl;
        }
    }

    motFile.close();

    ofstream STSvisFile(timestamp + "_stsp.vis");

    if (STSvisFile.is_open()) {
        STSvisFile << _sizeSTSp << "," << _sizeSTSp << endl;
        for (int i = 0; i < _sizeSTSp * _sizeSTSp; i++) {
            for (int j = 0; j < PERSPS; j++) {
                if (j == PERSPS - 1) {
                    STSvisFile << winRateSTSp_Visual[i][j];
                }
                else {
                    STSvisFile << winRateSTSp_Visual[i][j] << ",";
                }
            }
            if (i < _sizeSTSp * _sizeSTSp - 1) STSvisFile << endl;
        }
    }

    STSvisFile.close();

    ofstream STSmotFile(timestamp + "_stsp.mot");

    if (STSmotFile.is_open()) {
        STSmotFile << _sizeSTSp << "," << _sizeSTSp << endl;
        for (int i = 0; i < _sizeSTSp * _sizeSTSp; i++) {
            for (int j = 0; j < GRASPS; j++) {
                if (j == GRASPS - 1) {
                    STSmotFile << winRateSTSp_Motor[i][j];
                }
                else {
                    STSmotFile << winRateSTSp_Motor[i][j] << ",";
                }
            }
            if (i < _sizeSTSp * _sizeSTSp - 1) STSmotFile << endl;
        }
    }

    STSmotFile.close();
}

void ModelMNS::testBALData() {
    const string timestamp = to_string(time(nullptr));

    vector<Sequence*>* trainData = _data.permute();
    double** winRatePMC = init_test_buffer(trainData->size(), _sizePMC * _sizePMC);
    double*** winRateSTSp = init_test_buffer(trainData->size(), _sizeSTSp * _sizeSTSp, PERSPS);

    for(int i = 0; i < trainData->size(); i++) {
        for(int j = 0; j < trainData->at(i)->getMotorData()->size(); j++) {
            _msomMotor->activate(trainData->at(i)->getMotorData()->at(j));
        }
        for(int n = 0; n < _msomMotor->get_lattice()->getDim(); n++) {
            winRatePMC[i][n] = _msomMotor->get_output()->at(n);
        }

        _msomMotor->reset_context();
        for(int p = 0; p < PERSPS; p++) {
            for(int j = 0; j < trainData->at(i)->getVisualData(p)->size(); j++) {
                _msomVisual->activate(trainData->at(i)->getVisualData(p)->at(j));
            }
            for(int n = 0; n < _msomVisual->get_lattice()->getDim(); n++) {
                winRateSTSp[i][n][p] = _msomVisual->get_output()->at(n);
            }
            _msomVisual->reset_context();
        }
    }

    ofstream motFile(timestamp + "_pmc.act");

    if (motFile.is_open()) {
        motFile << _sizePMC << "," << _sizePMC << endl;
        for(int i = 0; i < trainData->size(); i++) {
            motFile << i << "," << trainData->at(i)->getGrasp() << ",";
            for (int j = 0; j < _sizePMC * _sizePMC; j++) {
                if (j == _sizePMC * _sizePMC - 1) {
                    motFile << winRatePMC[i][j] << endl;
                }
                else {
                    motFile << winRatePMC[i][j] << ",";
                }
            }
        }
    }

    motFile.close();

    ofstream STSvisFile(timestamp + "_stsp.act");

    if (STSvisFile.is_open()) {
        STSvisFile << _sizeSTSp << "," << _sizeSTSp << endl;

        for(int i = 0; i < trainData->size(); i++) {
            for(int p = 0; p < PERSPS; p++) {
                STSvisFile << i << "," << trainData->at(i)->getGrasp() << "," << p << ",";
                for (int j = 0; j < _sizeSTSp * _sizeSTSp; j++) {
                    if (j == _sizeSTSp * _sizeSTSp - 1) {
                        STSvisFile << winRateSTSp[i][j][p] << endl;
                    }
                    else {
                        STSvisFile << winRateSTSp[i][j][p] << ",";
                    }
                }
            }
        }
    }

    STSvisFile.close();

	free_test_buffer(winRatePMC, trainData->size(), _sizePMC * _sizePMC);
	free_test_buffer(winRateSTSp,trainData->size(), _sizeSTSp * _sizeSTSp, PERSPS);
}

double** ModelMNS::init_test_buffer(int p_size1, int p_size2) {
	double** result = new double*[p_size1];

	for(int i = 0; i < p_size1; i++) {
		result[i] = new double[p_size2];
	}

	return result;
}

double*** ModelMNS::init_test_buffer(int p_size1, int p_size2, int p_size3) {
	double*** result = new double**[p_size1];

	for (int i = 0; i < p_size1; i++) {
		result[i] = new double*[p_size2];
		for(int j = 0; j < p_size2; j++) {
			result[i][j] = new double[p_size3];
		}
	}

	return result;
}

void ModelMNS::free_test_buffer(double** p_buffer, int p_size1, int p_size2) {
	for (int i = 0; i < p_size1; i++) {
		delete p_buffer[i];
	}
	delete p_buffer;
}

void ModelMNS::free_test_buffer(double*** p_buffer, int p_size1, int p_size2, int p_size3) {
	for (int i = 0; i < p_size1; i++) {
		for (int j = 0; j < p_size2; j++) {
			delete p_buffer[i][j];
		}
		delete p_buffer[i];
	}
	delete p_buffer;
}


void ModelMNS::save() const {
    const string timestamp = to_string(time(nullptr));

    IOUtils::save_network(timestamp + "_pmc.json", _msomMotor);
    IOUtils::save_network(timestamp + "_stsp.json", _msomVisual);
}


void ModelMNS::load(string p_timestamp) {
    _msomMotor = (MSOM*)IOUtils::load_network(p_timestamp + "_pmc.json");
    _msomVisual = (MSOM*)IOUtils::load_network(p_timestamp + "_stsp.json");
}
