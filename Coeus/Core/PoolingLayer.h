#pragma once
#include "BaseLayer.h"

namespace Coeus
{
	class __declspec(dllexport) PoolingLayer : public BaseLayer
	{
	public:
		PoolingLayer(const string& p_id, int p_extent, int p_stride, initializer_list<int> p_in_dim = { 0 });
		PoolingLayer(PoolingLayer &p_copy, bool p_clone);
		~PoolingLayer();
		PoolingLayer* copy(bool p_clone) override;

		void init(vector<BaseLayer*>& p_input_layers, vector<BaseLayer*>& p_output_layers) override;

		void integrate(Tensor* p_input) override;
		void activate() override;

		void calc_derivative(map<string, Tensor*>& p_derivative) override;
		void calc_gradient(Gradient& p_gradient_map, map<string, Tensor*>& p_derivative_map) override;
		void reset() override;
		void copy_params(BaseLayer* p_source) override;

		Tensor* get_dim_tensor() override;
		json get_json() const override;

		
	private:
		int _extent;
		int _stride;
		vector<int> _max_index;
	};
}


