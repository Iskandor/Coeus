#pragma once
#include "SOM_analyzer.h"

namespace Coeus
{
	class __declspec(dllexport) Base_SOM_learning
	{
	public:
		Base_SOM_learning(SOM* p_som);
		virtual ~Base_SOM_learning();

		enum NEIGHBORHOOD_TYPE {
			EUCLIDEAN = 0,
			GAUSSIAN = 1
		};

		virtual void train(Tensor *p_input) = 0;		
		virtual void param_decay();

		SOM_analyzer* analyzer() const { return _som_analyzer; };

	protected:
		void init_training(double p_epochs);
		double calc_neighborhood(double p_d, NEIGHBORHOOD_TYPE p_type) const;
		double euclidean_distance(int p_x1, int p_y1, int p_x2, int p_y2) const;
		double gaussian_distance(double p_d, double p_sigma = 1) const;

		double _sigma0;
		double _sigma;
		double _lambda;
		double _iteration;

		SOM_analyzer* _som_analyzer;
		Tensor	_delta_w;
		Tensor	_dist_matrix;

	};
}


