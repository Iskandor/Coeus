#pragma once
#include "BaseLayer.h"
#include "NeuronOperator.h"
#include "TensorInitializer.h"

namespace Coeus
{
	class __declspec(dllexport) ConvLayer : public BaseLayer
	{
	public:
		ConvLayer(const string& p_id, ACTIVATION p_activation, TensorInitializer* p_initializer, int p_filters, int p_extent, int p_stride, int p_padding = 0, initializer_list<int> p_in_dim = {0});
		~ConvLayer();
		ConvLayer* clone() override;

		void init(vector<BaseLayer*>& p_input_layers) override;

		void integrate(Tensor* p_input) override;
		void activate() override;

		void calc_derivative(map<string, Tensor*>& p_derivative) override;
		void calc_delta(map<string, Tensor*>& p_delta_map, map<string, Tensor*>& p_derivative_map) override;
		void calc_gradient(map<string, Tensor>& p_gradient_map, map<string, Tensor*>& p_delta_map, map<string, Tensor*>& p_derivative_map) override;

		void override(BaseLayer* p_source) override;
		void reset() override;

		
		json get_json() const override;

	protected:
		Tensor* get_dim_tensor() override;

	private:
		int _filters;
		int _extent;
		int _stride;
		int _padding;

		NeuronOperator** _y;
		Param**			 _W;
		TensorInitializer *_initializer;
		IActivationFunction* _activation_function;

		Tensor* _filter_input;
	};
}
