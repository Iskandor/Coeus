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

			void update(SOM* p_som, const int p_winner);
			void end_epoch();

			double winner_diff(int p_size) const;
			double q_error() const { return _q_error; }

		private:
			double		_q_error;
			set<int>	_winner_set;
	};
}


