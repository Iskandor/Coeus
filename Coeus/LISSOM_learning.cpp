#include "LISSOM_learning.h"

using namespace Coeus;

LISSOM_learning::LISSOM_learning(LISSOM* p_som, LISSOM_params* p_params, SOM_analyzer* p_som_analyzer) : Base_SOM_learning(p_som, p_params, p_som_analyzer) {
	_lissom = p_som;
}

LISSOM_learning::~LISSOM_learning()
{
}

void LISSOM_learning::train(Tensor* p_input) {
	_lissom->activate(p_input);

	const int dim_lattice = _lissom->get_lattice()->get_dim();
	const int dim_input = _lissom->get_input_group<SimpleCellGroup>()->get_dim();

	Tensor* oi = _lissom->get_output();
	Tensor* in = _lissom->get_input_group<SimpleCellGroup>()->get_output();
	Tensor* wi = _lissom->get_afferent()->get_weights();
	Tensor* le = _lissom->get_latteral_e()->get_weights();
	Tensor* li = _lissom->get_latteral_i()->get_weights();

	const float alpha_a = static_cast<LISSOM_params*>(_params)->alpha_a();
	const float alpha_e = static_cast<LISSOM_params*>(_params)->alpha_e();
	const float alpha_i = static_cast<LISSOM_params*>(_params)->alpha_i();

	for (int i = 0; i < dim_lattice; i++) {
		for (int j = 0; j < dim_input; j++) {
			_delta_aw.set(i, alpha_a * oi->at(i) * in->at(j));
		}
	}

}
