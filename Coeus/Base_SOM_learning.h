#pragma once
#include "Base_SOM_params.h"
#include "SOM_analyzer.h"
#include <ppl.h>

using namespace concurrency;

namespace Coeus
{
	class __declspec(dllexport) Base_SOM_learning
	{
	public:
		Base_SOM_learning(SOM* p_som, Base_SOM_params* p_params, SOM_analyzer* p_som_analyzer);
		virtual ~Base_SOM_learning();

		enum NEIGHBORHOOD_TYPE {
			EUCLIDEAN = 0,
			GAUSSIAN = 1,
			ABS = 2
		};

		virtual void train(Tensor *p_input) = 0;		

		SOM_analyzer* analyzer() const { return _som_analyzer; };

		void set_mutex(critical_section* p_mutex) { _mutex = p_mutex; }

	protected:
		double calc_neighborhood(double p_d, NEIGHBORHOOD_TYPE p_type) const;

		Base_SOM_params* _params;
		SOM_analyzer* _som_analyzer;
		
		Tensor	_dist_matrix;

		critical_section* _mutex;

	};
}


