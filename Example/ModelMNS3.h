//
// Created by user on 14. 12. 2017.
//

#ifndef NEURONET_MODELMNS3_H
#define NEURONET_MODELMNS3_H

#include "Dataset.h"
#include "SOM.h"
#include "MSOM.h"
#include "MSOM_learning.h"

using namespace Coeus;

namespace MNS {

class ModelMNS3 {
public:
	ModelMNS3();
    ~ModelMNS3();

    void init();
    void run(int p_epochs);
    void save();
    void load(string p_timestamp);

    void testDistance();
    void testFinalWinners();
	void testMirror();

private:

	void activateF5(int p_index, MSOM* p_msom, vector<Tensor*>* p_input);
	void activateSTS(int p_index, MSOM* p_msom, vector<Tensor*>* p_input);
	void trainF5(int p_index, MSOM_learning* p_F5_learner, vector<Tensor*>* p_input);
	void trainSTS(int p_index, MSOM_learning* p_STS_learner, vector<Tensor*>* p_input);

    void prepareInputF5(int p_index, Tensor* p_input);
    void prepareInputSTS(int p_index, Tensor* p_input);

	void save_results(string p_filename, int p_dim_x, int p_dim_y, double* p_data, int p_category) const;

	static const int _sizeF5input = 16;
	static const int _sizeSTSinput = 40;
    static const int _sizeF5 = 12;
    static const int _sizeSTS = 16;
    static const int GRASPS = 3;
    static const int PERSPS = 4;

    Dataset _data;
    MSOM    *_F5;
    MSOM    *_STS;

	Tensor** _F5input;
    Tensor** _STSinput;

	int _f5_mask_pre[_sizeF5input + _sizeSTS * _sizeSTS];
	int _f5_mask_post[_sizeF5input + _sizeSTS * _sizeSTS];
	int _sts_mask[_sizeSTSinput + _sizeF5 * _sizeF5];
};

}

#endif //NEURONET_MODELMNS3_H