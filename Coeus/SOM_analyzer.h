#pragma once

#include <set>
#include "SOM.h"

using namespace std;

namespace Coeus {
	class __declspec(dllexport) SOM_analyzer
	{
		public:
			SOM_analyzer();
			~SOM_analyzer();

			void merge(vector<SOM_analyzer*> &p_analyzers);
			void update(SOM* p_som, const int p_winner);
			void create_umatrix(SOM* p_som);
			void end_epoch();

			double winner_diff(int p_size) const;
			double q_error(int p_dim = 0) const { return p_dim == 0 ? _q_error : _q_error / p_dim; }
			Tensor* umatrix() const { return _umatrix; }

			void save_umatrix(string p_filename);

		private:
			double		_q_error;
			set<int>	_winner_set;
			Tensor*		_umatrix;
	};
}


