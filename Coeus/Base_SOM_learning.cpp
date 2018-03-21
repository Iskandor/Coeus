#include "Base_SOM_learning.h"
#include "Metrics.h"

using namespace Coeus;

Base_SOM_learning::Base_SOM_learning(SOM* p_som, Base_SOM_params* p_params, SOM_analyzer* p_som_analyzer) {
	_som_analyzer = p_som_analyzer;
	_params = p_params;

	const int dim_lattice = p_som->get_lattice()->get_dim();

	_dist_matrix = Tensor::Zero({ dim_lattice, dim_lattice });

	int x1;
	int y1;
	int x2;
	int y2;

	for (int i = 0; i < dim_lattice; i++) {
		p_som->get_position(i, x1, y1);
		for (int j = 0; j < dim_lattice; j++) {
			p_som->get_position(j, x2, y2);
			_dist_matrix.set(i, j, Metrics::euclidean_distance(x1, y1, x2, y2));
		}
	}

}


Base_SOM_learning::~Base_SOM_learning()
{
}

double Base_SOM_learning::calc_neighborhood(const double p_d, const NEIGHBORHOOD_TYPE p_type) const {
	double result = 0;

	switch (p_type) {
	case EUCLIDEAN:
		result = 1.0 / p_d;
		break;
	case GAUSSIAN:
		result = Metrics::gaussian_distance(p_d, _params->sigma());
		break;
	}

	return result;
}

