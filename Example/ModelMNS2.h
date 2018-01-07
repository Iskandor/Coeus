//
// Created by user on 14. 12. 2017.
//

#ifndef NEURONET_MODELMNS2_H
#define NEURONET_MODELMNS2_H

#include "Dataset.h"
#include "SOM.h"
#include "MSOM.h"
#include "MSOM_learning.h"

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

	void preactivateF5(vector<Tensor*>* p_input);
	void preactivateSTS(vector<Tensor*>* p_input);
	void trainF5(MSOM_learning& p_F5_learner, vector<Tensor*>* p_input);
	void trainSTS(MSOM_learning& p_STS_learner, vector<Tensor*>* p_input);

    void prepareInputF5(Tensor* p_input);
    void prepareInputSTS(Tensor* p_input);
    void prepareInputPF();

	static const int _sizeF5input = 16;
	static const int _sizeSTSinput = 40;
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

	int _f5_mask[_sizeF5input + _sizePF * _sizePF];
	int _sts_mask[_sizeSTSinput + _sizePF * _sizePF];
};

}

#endif //NEURONET_MODELMNS2_H