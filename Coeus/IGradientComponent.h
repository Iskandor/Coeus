#pragma once

#include "Tensor.h"
#include "BaseLayer.h"
#include "LayerState.h"

using namespace FLAB;

namespace Coeus {
	
	class __declspec(dllexport) IGradientComponent
	{
	public:
		explicit IGradientComponent(BaseLayer* p_layer);
		virtual ~IGradientComponent();

		void set_delta(Tensor* p_delta);
		virtual void init() = 0;
		virtual void calc_deriv() = 0;
		virtual void calc_delta(Tensor* p_weights, Tensor* p_delta) = 0;
		virtual void calc_gradient(map<string, Tensor> &p_w_gradient, map<string, Tensor> &p_b_gradient);

		Tensor* get_output_deriv() { return &_deriv[_layer->_output_group->get_id()]; }
		Tensor* get_input_delta() { return &_delta[_layer->_input_group->get_id()]; }

	protected:
		void calc_deriv_group(BaseCellGroup* p_group);
		template<typename T>
		T*	get_layer() { return static_cast<T*>(_layer); }
		BaseLayer*	_layer;

		map<string, Tensor> _deriv;
		map<string, Tensor> _delta;
	};

}


