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

    void init(string p_timestamp = "");
    void run(int p_epochs);
	void save(string p_timestamp) const;

    void testDistance();
    void testFinalWinners();
	void testMirror();

private:
	void load(string p_timestamp);

    void prepareInputF5(Tensor* p_output, Tensor* p_input, MSOM* p_sts) const;
    void prepareInputSTS(Tensor* p_output, Tensor* p_input, MSOM* p_f5) const;

	static void save_results(string p_filename, int p_dim_x, int p_dim_y, double* p_data, int p_category);

	static const int _sizeF5input = 16;
	static const int _sizeSTSinput = 40;
    static const int GRASPS = 3;
    static const int PERSPS = 4;

    Dataset _data;
    MSOM    *_F5;
    MSOM    *_STS;

	int *_f5_mask_pre;
	int *_f5_mask_post;
	int *_sts_mask;
};

}

#endif //NEURONET_MODELMNS3_H