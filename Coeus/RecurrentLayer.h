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
	RecurrentLayer(RecurrentLayer& p_copy);
	explicit RecurrentLayer(const json& p_data);
	~RecurrentLayer();
	RecurrentLayer* clone() override;

	void activate() override;

	void calc_derivative(map<string, Tensor*>& p_derivative) override;
	void calc_delta(map<string, Tensor*>& p_delta_map, map<string, Tensor*>& p_derivative_map) override;
	void calc_gradient(map<string, Tensor>& p_gradient_map, map<string, Tensor*>& p_delta_map, map<string, Tensor*>& p_derivative_map) override;

	void override(BaseLayer* p_source) override;
	void reset() override;
	void init(vector<BaseLayer*>& p_input_layers) override;

	json get_json() const override;

private:
	explicit RecurrentLayer(RecurrentLayer* p_source);

	NeuronOperator* _y;

	Tensor* _context;
	Param* _W;

	TensorInitializer *_initializer;
};

}

