#pragma once

#include "Tensor.h"
#include "BaseLayer.h"
#include "NeuralNetwork.h"

using namespace FLAB;

namespace Coeus {
	
	class __declspec(dllexport) IGradientComponent
	{
	public:
		IGradientComponent(BaseLayer* p_layer, NeuralNetwork* p_network);
		virtual ~IGradientComponent();

		void set_delta(Tensor* p_delta);
		virtual void init() = 0;
		virtual void calc_deriv_estimate() = 0;
		virtual void calc_deriv() = 0;
		virtual void calc_delta(Tensor* p_weights, Tensor* p_delta) = 0;
		virtual void calc_gradient(map<string, Tensor> &p_gradient);
		virtual void reset();

		Tensor* get_output_deriv() { return &_deriv[_layer->_output_group->get_id()]; }
		Tensor* get_input_delta() { return &_delta[_layer->_input_group->get_id()]; }

	protected:
		void calc_deriv_group(BaseCellGroup* p_group);
		template<typename T>
		T*	get_layer() { return static_cast<T*>(_layer); }
		
		BaseLayer*		_layer;
		NeuralNetwork*	_network;

		map<string, Tensor> _deriv;
		map<string, Tensor> _delta;
	};

}


