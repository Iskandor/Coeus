#pragma once
#include "BaseLayer.h"
#include "Coeus.h"
#include "Param.h"
#include "IActivationFunction.h"
#include "NeuronOperator.h"
#include "TensorInitializer.h"

namespace Coeus
{
	class __declspec(dllexport) CoreLayer : public BaseLayer
	{
	public:
		CoreLayer(const string& p_id, int p_dim, ACTIVATION p_activation, TensorInitializer* p_initializer, int p_in_dim = 0);
		explicit CoreLayer(const json& p_data);
		CoreLayer(CoreLayer &p_copy);
		~CoreLayer();
		CoreLayer* clone() override;

		void activate() override;

		void calc_derivative(map<string, Tensor*>& p_derivative) override;
		void calc_gradient(map<string, Tensor>& p_gradient_map, map<string, Tensor*>& p_derivative_map) override;

		void override(BaseLayer* p_source) override;
		void reset() override {}
		void init(vector<BaseLayer*>& p_input_layers, vector<BaseLayer*>& p_output_layers) override;

		json get_json() const override;

	private:
		explicit CoreLayer(CoreLayer* p_source);
		Tensor* get_dim_tensor() override;

		NeuronOperator* _y;
		Param*			_W;		

		TensorInitializer *_initializer;
	};
}
