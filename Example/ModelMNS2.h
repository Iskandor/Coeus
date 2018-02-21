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

	void init(string p_timestamp = "");
	void run(int p_epochs);
	void save(string p_timestamp) const;

	void save_umatrix(string p_timestamp);
	void testDistance();
	void testFinalWinners();
	void testMirror(int p_persp);

private:
	void load(string p_timestamp);

	void prepareInputF5(Tensor* p_output, Tensor* p_input, SOM* p_pfg) const;
	void prepareInputSTS(Tensor* p_output, Tensor* p_input, SOM* p_pfg) const;
	void prepareInputPFG(Tensor* p_output, MSOM* p_f5, MSOM* p_sts) const;

	static void save_results(string p_filename, int p_dim_x, int p_dim_y, double* p_data, int p_category);

	static const int _sizeF5input = 16;
	static const int _sizeSTSinput = 40;
	static const int GRASPS = 3;
	static const int PERSPS = 4;

	Dataset _data;
	MSOM    *_F5;
	MSOM    *_STS;
	SOM		*_PFG;

	int *_f5_mask_pre;
	int *_f5_mask_post;
	int *_sts_mask;
};

}

#endif //NEURONET_MODELMNS2_H