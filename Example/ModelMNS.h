//
// Created by user on 5. 11. 2017.
//

#ifndef NEURONET_MNS_H
#define NEURONET_MNS_H

#include <MSOM.h>
#include "Dataset.h"

using namespace Coeus;

namespace MNS {

class ModelMNS {
public:
    ModelMNS();
    ~ModelMNS();

    void init();
    void run(int p_epochs);
    void save() const;
    void load(string p_timestamp);

    void testAllWinners();
    void testFinalWinners();
    void testDistance();
    void testBALData();

private:
	double** init_test_buffer(int p_size1, int p_size2);	
	double*** init_test_buffer(int p_size1, int p_size2, int p_size3);
	void free_test_buffer(double** p_buffer, int p_size1, int p_size2);
	void free_test_buffer(double*** p_buffer, int p_size1, int p_size2, int p_size3);

    static const int _sizePMC = 12;
	static const int _sizeSTSp = 16;
	static const int GRASPS = 3;
	static const int PERSPS = 4;

    Dataset _data;
    MSOM    *_msomMotor;
    MSOM    *_msomVisual;
};

}

#endif //NEURONET_MNS_H
