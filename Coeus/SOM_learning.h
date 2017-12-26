#pragma once
#include "SOM.h"
#include "SOM_analyzer.h"

namespace Coeus
{
	class __declspec(dllexport) SOM_learning
	{
	public:

		enum NEIGHBORHOOD_TYPE {
			EUCLIDEAN = 0,
			GAUSSIAN = 1
		};

		explicit SOM_learning(SOM* p_som);
		virtual ~SOM_learning();

		virtual void init_training(double p_alpha, double p_epochs);
		virtual void train(Tensor *p_input);
		virtual void param_decay();

		SOM_analyzer* analyzer() { return _som_analyzer; };


	private:
		double calc_neighborhood(double p_d, NEIGHBORHOOD_TYPE p_type) const;
		double euclidean_distance(int p_x1, int p_y1, int p_x2, int p_y2) const;
		double gaussian_distance(double p_d, double p_sigma = 1) const;

		double _sigma0;
		double _lambda;
		double _alpha0;
		double _sigma;
		double _alpha;
		double _iteration;

		SOM* _som;
		SOM_analyzer* _som_analyzer;
		Tensor	_delta_w;
		Tensor	_dist_matrix;
	};

}


