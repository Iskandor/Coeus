#pragma once
#include "SOM.h"

namespace Coeus {

class __declspec(dllexport) LSOM : public SOM
{
public:
	LSOM(string p_id, const int p_input_dim, const int p_dim_x, const int p_dim_y, const NeuralGroup::ACTIVATION p_activation);
	~LSOM();

	void activate(Tensor *p_input, Tensor* p_weights = nullptr) override;
	int find_winner(Tensor* p_input) override;

	Connection*  get_lattice_lattice() const { return _lattice_lattice; }

private:
	Tensor	_auxoutput;
	Connection* _lattice_lattice;
};

}


