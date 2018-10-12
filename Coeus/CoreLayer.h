#pragma once
#include "BaseLayer.h"
#include "Connection.h"

namespace Coeus {

class __declspec(dllexport) CoreLayer : public BaseLayer
{
	friend class CoreLayerGradient;
public:
	CoreLayer(const string& p_id, int p_dim, ACTIVATION p_activation);
	explicit CoreLayer(json p_data);
	CoreLayer(CoreLayer &p_copy);
	~CoreLayer();

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

