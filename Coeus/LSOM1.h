#pragma once
#include "SOM.h"

namespace Coeus {

class __declspec(dllexport) LSOM1 : public SOM
{
public:
	LSOM1(string p_id, const int p_input_dim, const int p_dim_x, const int p_dim_y, const NeuralGroup::ACTIVATION p_activation);
	~LSOM1();

	void activate(Tensor* p_input = nullptr) override;
	int find_winner(Tensor* p_input) override;

	Connection*  get_lateral() const { return _lateral; }

private:
	Tensor		_auxoutput;
	Connection* _lateral;
};

}


