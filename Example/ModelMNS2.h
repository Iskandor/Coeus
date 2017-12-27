//
// Created by user on 14. 12. 2017.
//

#ifndef NEURONET_MODELMNS2_H
#define NEURONET_MODELMNS2_H

#include "Dataset.h"
#include "SOM.h"
#include "MSOM.h"

using namespace Coeus;

namespace MNS {

class ModelMNS2 {
public:
    ModelMNS2();
    ~ModelMNS2();

    void init();
    void run(int p_epochs);
    void save();
    void load(string p_timestamp);

    void testDistance();
    void testFinalWinners();

private:
    void prepareInputF5(Tensor* p_input);
    void prepareInputSTS(Tensor* p_input);
    void prepareInputPF();

    static const int _sizeF5 = 12;
    static const int _sizeSTS = 16;
    static const int _sizePF = 14;
    static const int GRASPS = 3;
    static const int PERSPS = 4;

    Dataset _data;
    MSOM    *_F5;
    MSOM    *_STS;
    SOM     *_PF;

	Tensor _F5input;
    Tensor _STSinput;
	Tensor _PFinput;
};

}

#endif //NEURONET_MODELMNS2_H