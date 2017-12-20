#pragma once
#include "SOM.h"

namespace Coeus
{
	class SOM_learning
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

	private:
		double calc_neighborhood(int p_x1, int p_x2, int p_y1, int p_y2, NEIGHBORHOOD_TYPE p_type) const;
		double euclidean_distance(int p_x1, int p_y1, int p_x2, int p_y2) const;
		double gaussian_distance(double p_d, double p_sigma = 1) const;

		double _sigma;
		double _sigma0;
		double _lambda;
		double _alpha0;
		double _alpha;
		double _iteration;

		SOM* _som;
		Tensor _delta_w;
	};

}


