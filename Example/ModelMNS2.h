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
	void testMirror();

private:

	void activateF5(vector<Tensor*>* p_input);
	void activateSTS(vector<Tensor*>* p_input);
	void activatePF();
	void trainF5(MSOM_learning& p_F5_learner, vector<Tensor*>* p_input);
	void trainSTS(MSOM_learning& p_STS_learner, vector<Tensor*>* p_input);

    void prepareInputF5(Tensor* p_input);
    void prepareInputSTS(Tensor* p_input);
    void prepareInputPF();

	void save_results(string p_filename, int p_dim_x, int p_dim_y, double* p_data, int p_category) const;

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

	int _f5_mask_pre[_sizeF5input + _sizePF * _sizePF];
	int _f5_mask_post[_sizeF5input + _sizePF * _sizePF];
	int _sts_mask[_sizeSTSinput + _sizePF * _sizePF];
	int _pf_mask[_sizeF5 * _sizeF5 + _sizeSTS * _sizeSTS];
};

}

#endif //NEURONET_MODELMNS2_H