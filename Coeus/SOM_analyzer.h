#pragma once

#include <set>
#include "SOM.h"

using namespace std;

namespace Coeus {
	class __declspec(dllexport) SOM_analyzer
	{
		public:
			SOM_analyzer(SOM* p_som);
			~SOM_analyzer();

			void update(const int p_winner);
			void end_epoch();

			double winner_diff();
			double q_error() { return _q_error; }

		private:
			SOM*		_som;
			double		_q_error;
			set<int>	_winner_set;
	};
}


