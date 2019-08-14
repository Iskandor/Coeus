#pragma once
#include "BaseLayer.h"
#include "TensorInitializer.h"
#include "IActivationFunction.h"
#include "NeuronOperator.h"

namespace Coeus
{
	class __declspec(dllexport) GRULayer : public BaseLayer
	{
	public:
		GRULayer(const string& p_id, int p_dim, ACTIVATION p_activation, TensorInitializer* p_initializer, int p_in_dim = 0);
		~GRULayer();
		BaseLayer* clone() override;

		void init(vector<BaseLayer*>& p_input_layers, vector<BaseLayer*>& p_output_layers) override;
		void integrate(Tensor* p_input) override;
		void activate() override;

		void calc_derivative(map<string, Tensor*>& p_derivative) override;
		void calc_gradient(map<string, Tensor>& p_gradient_map, map<string, Tensor*>& p_derivative_map) override;

		void override(BaseLayer* p_source) override;
		void reset() override;

		
		json get_json() const override;

	private:
		Tensor* get_dim_tensor() override;

		NeuronOperator* _y;
		NeuronOperator* _h;
		NeuronOperator* _rg;
		NeuronOperator* _ug;

		Param*		_Why;
		Param*		_Wxh;
		Param*		_Wxrg;
		Param*		_Wxug;

		Tensor*		_reseted_input;
		Tensor*		_h_input;
		Tensor*		_state;

		TensorInitializer *_initializer;
	};
}


