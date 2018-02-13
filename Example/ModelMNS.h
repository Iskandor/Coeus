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

		void init(string p_timestamp = "");
		void run(int p_epochs);
		void save(string p_timestamp) const;
		void load(string p_timestamp);

		void save_umatrix(string p_timestamp);
		void testDistance();
		void testFinalWinners();


	private:
		static void save_results(string p_filename, int p_dim_x, int p_dim_y, double* p_data, int p_category);

		static const int _sizeF5input = 16;
		static const int _sizeSTSinput = 40;
		static const int GRASPS = 3;
		static const int PERSPS = 4;

		Dataset _data;
		MSOM    *_F5;
		MSOM    *_STS;
	};

}

#endif //NEURONET_MNS_H
