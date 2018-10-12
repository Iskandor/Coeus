#pragma once
#include "BaseLayer.h"

namespace Coeus {

class __declspec(dllexport) InputLayer : 	public BaseLayer
{
public:
	InputLayer(const string& p_id, int p_input_dim);
	explicit InputLayer(json p_data);
	InputLayer(InputLayer &p_copy);
	~InputLayer();

	void integrate(Tensor* p_input, Tensor* p_weights = nullptr) override;
	void activate(Tensor* p_input = nullptr) override;
	void override(BaseLayer* p_source) override;
	void reset() override {}
	void init(vector<BaseLayer*>& p_input_layers) override {}
	json get_json() const override;

private:
	SimpleCellGroup *_group;
};

}

