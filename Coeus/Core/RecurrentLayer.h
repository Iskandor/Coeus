#pragma once
#include "BaseLayer.h"
#include "Coeus.h"
#include "Param.h"
#include "NeuronOperator.h"
#include "TensorInitializer.h"

namespace Coeus {

class __declspec(dllexport) RecurrentLayer : public BaseLayer
{
public:
	RecurrentLayer(const string& p_id, int p_dim, ACTIVATION p_activation, TensorInitializer* p_initializer, int p_in_dim = 0);
	RecurrentLayer(RecurrentLayer& p_copy, bool p_clone);
	explicit RecurrentLayer(const json& p_data);
	~RecurrentLayer();
	RecurrentLayer* copy(bool p_clone) override;

	void activate() override;

	void calc_derivative(map<string, Tensor*>& p_derivative) override;
	void calc_gradient(Gradient& p_gradient_map, map<string, Tensor*>& p_derivative_map) override;

	void reset() override;
	void copy_params(BaseLayer* p_source) override;
	void init(vector<BaseLayer*>& p_input_layers, vector<BaseLayer*>& p_output_layers) override;

	json get_json() const override;

private:
	Tensor* get_dim_tensor() override;

	
	NeuronOperator* _y;

	Tensor* _context;
	Param* _W;

	TensorInitializer *_initializer;
};

}

